from __future__ import annotations
import pprint
from typing import Callable, List, Tuple, Dict, cast, Union, Optional, TypeVar, Generic
import functools, itertools, operator
from tinygrad.nn.state import get_parameters
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, merge_dicts, getenv, all_int, Context, GRAPH, flatten, GraphException
from tinygrad.device import BufferCopy, Compiled, JITRunner, CompiledASTRunner, Buffer, Device
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sint
from weakref import ref, WeakKeyDictionary
from dataclasses import dataclass
import pickle

@dataclass(frozen=True)
class JitItem:
  prg: JITRunner  # or a graph executor like MetalGraph
  rawbufs: List[Optional[Buffer]]

def get_jit_stats(jit_cache: List[JitItem]) -> Tuple[sint, int]:
  return functools.reduce(operator.add, [ji.prg.op_estimate for ji in jit_cache if isinstance(ji.prg, CompiledASTRunner)], 0), \
         functools.reduce(operator.add, [ji.prg.mem_estimate for ji in jit_cache if isinstance(ji.prg, CompiledASTRunner)], 0)
def get_input_replace(jit_cache: List[JitItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.rawbufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace
def get_jc_idxs_with_updatable_launch_dims(jit_cache: List[JitItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledASTRunner) and ((ji.prg.global_size and not all_int(ji.prg.global_size)) or (ji.prg.local_size and not all_int(ji.prg.local_size)))]  # noqa: E501
def get_jc_idxs_with_updatable_var_vals(jit_cache: List[JitItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledASTRunner) and ji.prg.vars]

def apply_graph_to_jit(jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]) -> List[JitItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  graphed_jit_cache: List[JitItem] = []
  current_batch: List[JitItem] = []
  current_device: Union[Compiled, None] = None

  # Flush the current batch.
  def flush():
    nonlocal current_batch, current_device
    assert current_device is not None
    try:
      if len(current_batch) <= 1: raise GraphException("only one kernel doesn't graph")
      graphed_jit_cache.append(JitItem(current_device.graph(current_batch, input_rawbuffers, var_vals), cast(List[Optional[Buffer]], input_rawbuffers))) # noqa: E501
      if DEBUG >= 2: print(f"\tJIT GRAPHing batch with {len(current_batch)} kernels on device {current_device}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"\tJIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_device}: {e}")
    current_batch = []
    current_device = None

  for i,ji in enumerate(jit_cache):
    # If the jit item can potentially be graphed, put it in a batch.
    can_be_graphed = isinstance(ji.prg, CompiledASTRunner) and ji.prg.device.graph
    if can_be_graphed:
      assert isinstance(ji.prg, CompiledASTRunner)
      # If the device changed we flush the batch early and append this item for the next batch.
      if current_device is not None and ji.prg.device != current_device: flush()
      current_device = ji.prg.device
      current_batch.append(ji)

    # The flush is done when (1) ji is the last one, (2) the size of batch exceeds the maximum batch size or
    # (3) the current jit item cannot be graphed, so the current batch is flushed before such a jit item is added.
    if len(current_batch) > 0 and (i==len(jit_cache)-1 or len(current_batch) >= getenv("JIT_BATCH_SIZE", 64) or not can_be_graphed): flush()

    # If the jit item cannot be graphed, put it right into the final cache after the flush.
    if not can_be_graphed: graphed_jit_cache.append(ji)
  return graphed_jit_cache

# *** JIT ***

@dataclass(frozen=True)
class SerializedTensor:
  device: str
  shape: Tuple[int, ...]
  dtype: DType
  replace: Dict[Tuple[int, int], int]

GLOBAL_CACHE = True

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()
    if GLOBAL_CACHE:
      import inspect
      import os
      line = inspect.getsourcelines(self.fxn)[1]
      file = inspect.getsourcefile(self.fxn)
      filename = f"{file}_{line}.pkl"
      path = os.path.join("/tmp/tinygrad_jitcache", filename)
      if os.path.exists(path):
        self.load(path)


  def reset(self):
    self.jit_cache: List[JitItem] = []
    self.input_replace: Dict[Tuple[int, int], int] = {}
    self.cnt: int = 0
    self.ret: Optional[ReturnType] = None
    self.expected_vals: Optional[Tuple[Variable, ...]] = None
    self.expected_name_sts_dtype_device: Optional[Tuple[Tuple[Union[int, str], ShapeTracker, DType, Union[str, Tuple[str, ...]]], ...]] = None

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> ReturnType:
    # all inputs (except const) are realized
    input_tensors: Dict[Union[int, str], Union[LazyBuffer, MultiLazyBuffer]] = { cast(Union[int, str], k):v.realize().lazydata for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor }  # noqa: E501
    expected_name_sts_dtype_device = tuple([(k, v.st.unbind()[0] if isinstance(v, LazyBuffer) else ShapeTracker.from_shape(v.shape), v.dtype, v.device) for k,v in input_tensors.items()]) #noqa: E501

    # get rawbuffers
    lbs: List[LazyBuffer] = [v for v in input_tensors.values() if isinstance(v, LazyBuffer)] + flatten([mlb.lbs for mlb in input_tensors.values() if isinstance(mlb, MultiLazyBuffer)]) #noqa: E501
    input_rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
    assert len(set(input_rawbuffers)) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.st.var_vals for arg in lbs] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])  # noqa: E501
    expected_vals = tuple(var_vals.keys())

    if self.cnt >= 2:
      # jit exec
      assert self.expected_vals == expected_vals and self.expected_name_sts_dtype_device is not None, "missing/mismatch of var_vals"
      assert all(x[0] == y[0] and x[1].views == y[1].views and x[2] == y[2] and x[3] == y[3]
                 for x,y in zip(self.expected_name_sts_dtype_device, expected_name_sts_dtype_device)), \
        f"mismatch of input tensors, expected {self.expected_name_sts_dtype_device} got {expected_name_sts_dtype_device}"
      for (j,i),input_idx in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_idx]
      for ji in self.jit_cache: ji.prg(cast(List[Buffer], ji.rawbufs), var_vals, wait=DEBUG>=2, jit=True)
    elif self.cnt == 1:
      # jit capture
      self.expected_vals, self.expected_name_sts_dtype_device = expected_vals, expected_name_sts_dtype_device
      CacheCollector.start(var_vals)
      with Context(GRAPH=getenv("JITGRAPH", GRAPH.value)):
        self.ret = self.fxn(*args, **kwargs)
        for p in get_parameters(self.ret): p.realize()
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1 and len(set(get_input_replace(self.jit_cache, input_rawbuffers).values())) != len(input_rawbuffers):
        print("WARNING: some input tensors not found")
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # Condense the items into a graph executor.
      if getenv("JIT") != 2: self.jit_cache = apply_graph_to_jit(self.jit_cache, input_rawbuffers, var_vals)

      self.input_replace = get_input_replace(self.jit_cache, input_rawbuffers)

      if GLOBAL_CACHE:
        import inspect
        import os
        line = inspect.getsourcelines(self.fxn)[1]
        file = inspect.getsourcefile(self.fxn)
        filename = f"{file}_{line}.pkl"
        path = os.path.join("/tmp/tinygrad_jitcache", filename)
        self.save(path)

    elif self.cnt == 0:
      # jit ignore
      self.ret = self.fxn(*args, **kwargs)
      for p in get_parameters(self.ret): p.realize()

    # clear jit inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None

    self.cnt += 1
    return cast(ReturnType, self.ret)
  
  def load(self, path:str):
    self.reset()
    with open(path, "rb") as f: 
      sjc, self.input_replace, self.expected_vals, self.expected_name_sts_dtype_device, ret = pickle.load(f)

      for prg, bufs in sjc:
        # Deserialize prog
        if isinstance(prg, tuple):
          name, src, lib, device, gs, ls = prg
          prg = CompiledASTRunner(None, name, src, Device[device], gs, ls, lib)
        elif isinstance(prg, BufferCopy):
          continue
        else:
          raise NotImplementedError()
        
        rawbufs = []
        for b in bufs:
          if b is not None:
            r = Buffer(b[0], b[1], b[2], options=b[3])
            rawbufs.append(r)
          else:
            rawbufs.append(None)
          
        self.jit_cache.append(JitItem(prg, rawbufs))
      
      # Deserialize ret
      # TODO: maybe can abstract this into some kind of 'transform' helper, that just applies a transformation on the child types
      def deserialize(obj):
        if isinstance(obj, list):
          return tuple(deserialize(x) for x in obj)
        if isinstance(obj, tuple):
          return [deserialize(x) for x in obj]
        elif isinstance(obj, dict):
          return {k: deserialize(v) for k,v in obj.items()}
        elif isinstance(obj, SerializedTensor):
          lbs = []
          for (j, i) in obj.replace:
            raw = self.jit_cache[j].rawbufs[i]
            lb = LazyBuffer(obj.device, ShapeTracker.from_shape(obj.shape), obj.dtype)
            lb.realized = raw
            lbs.append(lb)

          # TODO: fix multi
          assert len(lbs) == 1 
          lazydata = lbs[0] # if len(lbs) == 1 else MultiLazyBuffer(lbs, obj.axis, obj.real)
          return Tensor(lazydata, obj.device)
        else:
          raise TypeError(f"Invalid return type for the jit {type(obj)}")

      self.ret = deserialize(ret)
      # pprint.pprint(self.jit_cache)
      # pprint.pprint(sjc)
      
    self.cnt = 2
    
  def save(self, path:str):
    if self.jit_cache is None:
      raise RuntimeError("Cache must be initialized before saving")
    
    sjc = []
    for j, ji in enumerate(self.jit_cache):
      # Serialize prog
      # TODO: handle graph
      if isinstance(p := ji.prg, CompiledASTRunner):
        sji = (p.display_name, p.prg, p.lib, p.device.dname, p.global_size, p.local_size)
      elif isinstance(ji.prg, BufferCopy):
        sji = ji
      else:
        raise NotImplementedError(f"{ji.prg}")

      # Serialize bufs 
      # TODO: maybe move this to new load/save methods inside Buffer?
      bufs = []
      for i, b in enumerate(ji.rawbufs):
        if (j, i) in self.input_replace:
          # assert b is None, b
          # continue
          pass
        if b is not None:
          assert not b.is_opaque, "Opaque buffer cannot be serialized"
          sb = (b.device, b.size, b.dtype, b.options)
        else:
          sb = None
        bufs.append(sb)

      sjc.append((sji, bufs))

    # Ret
    def serialize(obj):
      if isinstance(obj, list):
        return tuple(serialize(x) for x in obj)
      if isinstance(obj, tuple):
        return [serialize(x) for x in obj]
      elif isinstance(obj, dict):
        return {k: serialize(v) for k,v in obj.items()}
      elif isinstance(obj, Tensor):
        lbs: List[LazyBuffer]
        if isinstance(obj.lazydata, LazyBuffer):
          lbs = [obj.lazydata] 
        elif isinstance(obj.lazydata, MultiLazyBuffer):
          lbs = obj.lazydata.lbs
        else:
          assert False
        
        rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
        replace: List[Tuple[int, int]] = []
        for j,ji in enumerate(self.jit_cache):
          for i,a in enumerate(ji.rawbufs):
            if a in rawbuffers:
              replace.append((j, i))
        return SerializedTensor(device=obj.device, shape=obj.shape, dtype=obj.dtype, replace=replace)
      else:
        raise TypeError(f"Invalid return type for the jit {type(obj)}")

    ret = serialize(self.ret)

    # pprint.pprint(self.jit_cache)
    # pprint.pprint(sjc)
    # pprint.pprint(self.ret)

    with open(path, "wb") as f: 
      pickle.dump((
        sjc,
        self.input_replace, 
        self.expected_vals, 
        self.expected_name_sts_dtype_device,
        ret,
      ), f)


class PlaceHolder:
  def __init__(self, buf:Buffer):
    self.size, self.dtype, self.device, self.ref, self.bufid, self.options = buf.size, buf.dtype, buf.device, ref(buf), id(buf._buf), buf.options
  def to_tuple(self): return (self.size, self.dtype, self.device, self.bufid, self.options)
  def __hash__(self): return hash(self.to_tuple())
  def __eq__(self, x): return isinstance(x, PlaceHolder) and self.to_tuple() == x.to_tuple()
  def alloc_if_needed(self, buffer_cache: Dict[PlaceHolder, Buffer]) -> Buffer:
    ret = self.ref()
    if ret: return ret
    if self not in buffer_cache: buffer_cache[self] = Buffer(self.device, self.size, self.dtype, options=self.options)
    return buffer_cache[self]

class _CacheCollector:
  def __init__(self):
    self.cache: Optional[List[Tuple[JITRunner, List[Union[Buffer, PlaceHolder]]]]] = None

  def start(self, var_vals:Optional[Dict[Variable, int]]=None):
    self.cache = []
    self.placeholders: WeakKeyDictionary[Buffer, PlaceHolder] = WeakKeyDictionary()
    self.var_vals = var_vals if var_vals is not None else {}

  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    for k,v in var_vals.items(): assert k in self.var_vals and self.var_vals[k] == v, f"var_vals {k} mismatch {v} != {self.var_vals.get(k)}"
    # NOTE: this is making an assumption that 0 is special
    # TODO: this is wrong for sync and wait
    if len(rawbufs): self.placeholders[rawbufs[0]] = PlaceHolder(rawbufs[0])
    self.cache.append((prg, [self.placeholders.get(x, x) if isinstance(x, Buffer) else x for x in rawbufs]))

  def finish(self) -> List[JitItem]:
    if self.cache is None: return []
    buffer_cache: Dict[PlaceHolder, Buffer] = {}
    saved_cache, self.cache = self.cache, None
    return [JitItem(prg, [x.alloc_if_needed(buffer_cache) if isinstance(x, PlaceHolder) else x for x in pl]) for prg, pl in saved_cache]
CacheCollector = _CacheCollector()
