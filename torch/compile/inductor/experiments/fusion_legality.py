"""真实融合谓词的受控协议与真实 CPU 编译；不执行 GPU，不给虚构时长。"""
from pathlib import Path
import os,json,types
R=Path(__file__).resolve().parents[1];O=R/'experiments/fusion-legality';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'experiments/cache-fusion-legality');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch,sympy
from torch.utils._ordered_set import OrderedSet as OS
from torch._inductor import config,metrics,ir
from torch._inductor.scheduler import Scheduler,FusedSchedulerNode,SchedulerNode,BaseScheduling,WhyNoFuse
from torch._inductor.dependencies import MemoryDep,StarDep,WeakDep,ReadWrites
from torch._inductor.virtualized import V
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch.fx.experimental.proxy_tensor import make_fx
I=sympy.Symbol('c0',integer=True);N=sympy.Integer(6)
def md(name,index=I,size=(N,),mode=None):return MemoryDep(name,index,(I,)+tuple(sympy.Symbol(f'c{k}',integer=True) for k in range(1,len(size))),size,mode)
class Node:
 def __init__(self,name='a',device='cpu',anc=(),reads=(),writes=(),unmet=(),reduction=False,template=False,order=0,group=(6,1),split=False):
  self.name=name;self.device=torch.device(device);self.ancestors=OS(anc);self.read_writes=ReadWrites(OS(reads),OS(writes),OS());self.unmet_dependencies=OS(unmet);self.reduction=reduction;self.template=template;self.min_order=self.max_order=order;self.group=(self.device,tuple(map(sympy.Integer,group)));self.split=split;self.node=types.SimpleNamespace()
 def get_name(self):return self.name
 def get_operation_names(self):return OS([self.name])
 def get_buffer_names(self):return OS(d.name for d in self.read_writes.writes)
 def get_device(self):return self.device
 def get_nodes(self):return [self]
 def is_template(self):return self.template
 def is_reduction(self):return self.reduction
 def is_foreach(self):return False
 def has_aliasing_or_mutation(self):return False
 def is_split_scan(self):return self.split
 def get_ranges(self):return ([N],[])
class Backend:
 def __init__(self,allow=True):self.events=[];self.allow=allow
 def can_fuse_horizontal(self,a,b):self.events.append('horizontal');return self.allow
 def can_fuse_vertical(self,a,b):self.events.append('vertical');return self.allow

def sched(shared=1,allow=True):
 s=object.__new__(Scheduler);s.mutation_renames={};s.mutation_real_name={};s.name_to_buf={};s.name_to_fused_node={};s.score_fusion_memory=lambda a,b:shared;s.has_shared_data_after_reordering_loop=lambda a,b:False
 backend=Backend(allow);s.get_backend=lambda dev:backend
 return s,backend
results={'protocol':{},'dependencies':{},'cycles':{},'simd':{},'cpu':{}}
reasons=[];original_why=WhyNoFuse.__call__
def why(self,reason,*args):reasons.append(reason % args if args else reason);return original_why(self,reason,*args)
WhyNoFuse.__call__=why
try:
 with V.set_graph_handler(types.SimpleNamespace(no_fuse_buffer_names=set())),config.patch({'max_fusion_size':64,'aggressive_fusion':False,'loop_ordering_after_fusion':False}):
  def trial(label,a,b,expected,patch=None,shared=1,allow=True,setup=None):
   s,be=sched(shared,allow)
   if setup:setup(s)
   reasons.clear()
   with config.patch(patch or {}):actual=s.can_fuse(a,b)
   assert bool(actual)==expected,(label,actual,reasons)
   results['protocol'][label]={'allowed':bool(actual),'backend_calls':be.events,'reasons':list(reasons)}
  trial('horizontal',Node('a'),Node('b'),True)
  trial('backend_veto',Node('a'),Node('b'),False,allow=False)
  a=Node('a',writes=[md('buf')]);b=Node('b',anc=['a'],unmet=[md('buf')])
  trial('vertical',a,b,True)
  trial('reverse',b,a,False)
  trial('different_device',Node('a'),Node('b',device='cuda:0'),False)
  trial('max_size',Node('a'),Node('b'),False,patch={'max_fusion_size':1})
  V.graph.no_fuse_buffer_names.add('buf')
  trial('explicit_no_fuse',a,b,False)
  V.graph.no_fuse_buffer_names.clear()
  trial('template_as_consumer',Node('a'),Node('b',template=True),False)
  trial('template_reduction_epilogue',Node('a',template=True),Node('b',reduction=True),False)
  trial('offset_mismatch',a,Node('b',anc=['a'],unmet=[md('buf',I+1)]),False)
  trial('star_unmatched',a,Node('b',anc=['a'],unmet=[StarDep('buf')]),False)
  def intermediate(s):
   s.name_to_buf['midbuf']=types.SimpleNamespace(defining_op=Node('middle'))
   s.name_to_fused_node['middle']=Node('middle',anc=['a'])
  trial('intermediate',a,Node('b',anc=['a','middle'],unmet=[md('buf'),md('midbuf')]),False,setup=intermediate)
  trial('peak_distance_64',Node('a',order=0),Node('b',order=64),True)
  trial('peak_distance_65',Node('a',order=0),Node('b',order=65),False)
  trial('no_shared',Node('a'),Node('b'),False,shared=0)
  trial('aggressive_no_shared',Node('a'),Node('b'),True,shared=0,patch={'aggressive_fusion':True})
  trial('aggressive_reduction',Node('a',reduction=True),Node('b'),False,shared=0,patch={'aggressive_fusion':True})
  n=Node('a');trial('same_node',n,n,False)
  # Real dependency dataclasses + actual matcher, no fabricated tensor values.
  s,_=sched();pairs=[('exact',md('buf'),md('buf'),True),('shift',md('buf',I+1),md('buf'),False),('broadcast_prefix',md('buf',I,(N,sympy.Integer(3))),md('buf'),True),('star',StarDep('buf'),md('buf'),False),('mode_star',StarDep('buf','atomic_add'),md('buf',mode='atomic_add'),True),('mode_memory_early_return',md('other',I+1,mode='atomic_add'),md('buf',mode='atomic_add'),True)]
  for name,read,write,expected in pairs:
   actual=s.fusable_read_and_write(read,write);assert actual==expected;results['dependencies'][name]={'allowed':actual,'scope':'actual matcher + real dependencies; mode special case is not a general cross-buffer fusion claim'}
  s.mutation_renames['old']='buf';assert s.fusable_read_and_write(md('old'),md('buf'));results['dependencies']['renamed']={'allowed':True}
  tmp=sympy.Symbol('tmp0',integer=True)
  assert not s.fusable_read_and_write(md('buf',tmp),md('buf',tmp));results['dependencies']['indirect']={'allowed':False}
  s.mutation_real_name['new']='old'
  weak=WeakDep('read_result','new');producer=Node('p',reads=[md('old')],writes=[md('read_result')]);consumer=Node('q',writes=[md('new')])
  assert s.fusable_weak_dep(weak,producer,consumer)
  consumer.read_writes.writes=OS([md('new',I+1)])
  assert not s.fusable_weak_dep(weak,producer,consumer)
  results['dependencies']['weak']={'same_index':True,'different_index':False}
  # Actual cycle finder, actual FusedSchedulerNode subclass with explicit metadata.
  class Group(FusedSchedulerNode):
   def __init__(self,names,anc):self.names=OS(names);self.ancestors=OS(anc)
   def get_operation_names(self):return self.names
   def get_name(self):return '_'.join(self.names)
  a=Node('A');d=Node('D',anc=['C']);s,_=sched();s.name_to_fused_node={'C':Group(['B','C'],['A'])}
  assert s.will_fusion_create_cycle(a,d)
  s.name_to_fused_node={'C':Group(['B','C'],[])};assert not s.will_fusion_create_cycle(a,d)
  results['cycles']={'new_path':True,'no_new_path':False,'scope':'actual DFS with metadata-only fused groups, not a full scheduling run'}
  # SIMD decision invokes real code; only select_tiling is a controlled provider.
  simd=SIMDScheduling(None);simd.select_tiling=lambda nodes,*args:(sympy.Integer(6),sympy.Integer(1))
  for name,n1,n2,expect in [
   ('pointwise_equal',Node('p'),Node('q'),True),('pointwise_mismatch',Node('p'),Node('q',group=(7,1)),False),
   ('reduction_equal',Node('p',reduction=True,group=(2,3)),Node('q',reduction=True,group=(2,3)),True),
   ('reduction_mismatch',Node('p',reduction=True,group=(2,3)),Node('q',reduction=True,group=(3,2)),False),
   ('split_scan_reason_only',Node('p',reduction=True,group=(2,3),split=True),Node('q',reduction=True,group=(2,3)),True)]:
   reasons.clear();actual=simd.can_fuse(n1,n2);assert bool(actual)==expect;results['simd'][name]={'allowed':bool(actual),'reasons':list(reasons),'scope':'actual SIMD predicate + metadata fixtures; no Triton execution'}
  # Existing CUDACombined code path reaches inherited NotImplementedError.
  combined=CUDACombinedScheduling(None);template=object.__new__(SchedulerNode);template.node=object.__new__(ir.CUDATemplateBuffer)
  try:combined.can_fuse_horizontal(template,Node('epilogue'))
  except NotImplementedError:results['cuda_cpp_horizontal_boundary']={'error':'NotImplementedError','scope':'direct backend hook with metadata-only template; no real template compilation'}
  else:raise AssertionError('expected inherited missing hook')
  for name in ['can_fuse_vertical','can_fuse_horizontal']:
   try:getattr(BaseScheduling(),name)(Node('a'),Node('b'))
   except NotImplementedError:pass
   else:raise AssertionError('BaseScheduling hook implemented unexpectedly')
  results['base_hooks']={'vertical':'NotImplementedError','horizontal':'NotImplementedError'}
  # CPU benchmark_fusion enabled still takes its explicit early-return branch.
  s,_=sched();s.benchmark_fused_nodes=lambda nodes:(_ for _ in ()).throw(AssertionError('must not benchmark CPU'))
  with config.patch({'benchmark_fusion':True}):assert s.speedup_by_fusion(Node('a'),Node('b'))
  results['cpu_benchmark_bypass']={'allowed':True,'benchmark_called':False}
finally:WhyNoFuse.__call__=original_why
# Complete CPU compilation/execute; default backend vs explicit veto of each hook.
orig_h=CppScheduling.can_fuse_horizontal;orig_v=CppScheduling.can_fuse_vertical;orig_fuse=Scheduler.fuse_nodes;orig_save=GraphLowering.save_output_code
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
def vertical(x):
 a=torch.exp(x);return (a,a+1)
cases=[('horizontal_allow',lambda x:(x+1,x*2),False),('horizontal_veto',lambda x:(x+1,x*2),True),('vertical_allow',vertical,False),('vertical_veto',vertical,True)]
try:
 for label,fn,veto in cases:
  state={'calls':[]}
  def h(self,a,b):
   r=False if veto else orig_h(self,a,b);state['calls'].append({'hook':'horizontal','allowed':bool(r)});return r
  def v(self,a,b):
   r=False if veto else orig_v(self,a,b);state['calls'].append({'hook':'vertical','allowed':bool(r)});return r
  def fuse(self,nodes):
   state['before']=[n.get_name() for n in nodes];out=orig_fuse(self,nodes);state['after']=[{'kind':type(n).__name__,'name':n.get_name()} for n in out];return out
  CppScheduling.can_fuse_horizontal=h;CppScheduling.can_fuse_vertical=v;Scheduler.fuse_nodes=fuse
  GraphLowering.save_output_code=staticmethod(lambda code:(O/f'{label}_output.py').write_text(code))
  gm=make_fx(fn)(x);metrics.reset()
  with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False,'max_fusion_size':64}):compiled=compile_fx_inner(gm,[x],is_inference=True)
  actual_input=x.clone();before=actual_input.clone();actual=compiled([actual_input]);expected=fn(x)
  torch.testing.assert_close(actual,expected);assert torch.equal(actual_input,before)
  assert all(o.untyped_storage().data_ptr()!=actual_input.untyped_storage().data_ptr() for o in actual)
  assert len(state['after'])==(2 if veto else 1),(label,state)
  state.update(outputs=[v.tolist() for v in actual],generated_kernel_count=metrics.generated_kernel_count,input_preserved=True,outputs_do_not_alias_input=True,scope='real full CPU compilation/execution; hook veto is an explicit experimental policy, not a registered new device')
  results['cpu'][label]=state
finally:
 CppScheduling.can_fuse_horizontal=orig_h;CppScheduling.can_fuse_vertical=orig_v;Scheduler.fuse_nodes=orig_fuse;GraphLowering.save_output_code=staticmethod(orig_save)
results['scope']='protocol fixtures do not prove GPU execution or speed; full CPU runs validate actual consumer/backend integration'
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
