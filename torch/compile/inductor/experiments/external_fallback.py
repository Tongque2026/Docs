"""真实 fallback 编译与运行：结构参数、布局约束、mutation、alias；不执行新设备。"""
from pathlib import Path
import os,json,collections,types
R=Path(__file__).resolve().parents[1];O=R/'experiments/external-fallback';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'experiments/cache-external-fallback');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch._inductor import config,ir,lowering
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
counts=collections.Counter();runtime_events=[]
lib=torch.library.Library('inductor_external_lesson','DEF')
lib.define('pack(Tensor[] xs, int dim, *, float alpha=2., bool reverse=False) -> (Tensor, Tensor[])')
lib.define('contiguous(Tensor x) -> Tensor')
lib.define('bump(Tensor(a!) x, float delta) -> ()')
lib.define('bump_copy(Tensor(a!) x, float delta) -> ()')
lib.define('view(Tensor(a) x) -> Tensor(a)')
lib.define('bad(Tensor(a!) x) -> Tensor(a!)')
def mark(name,x):counts[name+'_cpu']+=1;runtime_events.append({'op':name,'stride':list(x.stride()),'device':str(x.device)})
def pack_cpu(xs,dim,*,alpha=2.,reverse=False):
 mark('pack',xs[0]);out=xs[0]+alpha*xs[1];parts=[v.clone() for v in (xs[::-1] if reverse else xs)];return out,parts
def contiguous_cpu(x):mark('contiguous',x);return (x*2).contiguous()
def bump_cpu(x,delta):mark('bump',x);x.add_(delta)
def bump_copy_cpu(x,delta):mark('bump_copy',x);x.add_(delta)
def view_cpu(x):mark('view',x);return x.view_as(x)
def bad_cpu(x):mark('bad',x);return x.add_(1)
for name,fn in [('pack',pack_cpu),('contiguous',contiguous_cpu),('bump',bump_cpu),('bump_copy',bump_copy_cpu),('view',view_cpu),('bad',bad_cpu)]:lib.impl(name,fn,'CPU')
@torch.library.register_fake('inductor_external_lesson::pack')
def pack_fake(xs,dim,*,alpha=2.,reverse=False):
 counts['pack_fake']+=1;parts=xs[::-1] if reverse else xs;return torch.empty_like(xs[0]),[torch.empty_like(x) for x in parts]
@torch.library.register_fake('inductor_external_lesson::contiguous')
def contiguous_fake(x):counts['contiguous_fake']+=1;return torch.empty_like(x,memory_format=torch.contiguous_format)
@torch.library.register_fake('inductor_external_lesson::bump')
def bump_fake(x,delta):counts['bump_fake']+=1
@torch.library.register_fake('inductor_external_lesson::bump_copy')
def bump_copy_fake(x,delta):counts['bump_copy_fake']+=1
@torch.library.register_fake('inductor_external_lesson::view')
def view_fake(x):counts['view_fake']+=1;return x.view_as(x)
@torch.library.register_fake('inductor_external_lesson::bad')
def bad_fake(x):counts['bad_fake']+=1;return x
op=torch.ops.inductor_external_lesson
# Explicit registration happens only inside this isolated interpreter.
for name in ['pack','bump','view','bad']:lowering.make_fallback(getattr(op,name).default)
for name in ['contiguous','bump_copy']:lowering.make_fallback(getattr(op,name).default,layout_constraint=lowering.require_contiguous)
x=torch.tensor([[-2.,0.,3.],[1.,-1.,2.]]);y=torch.ones_like(x)
xs=torch.arange(6,dtype=torch.float32).reshape(3,2).t()
def packed(x,y):
 a,parts=op.pack.default([x+1,y*2],0,alpha=3.,reverse=True);return (a.relu(),parts[0],parts[1])
def plain(x):return (op.contiguous.default(x),)
def mutate(x):op.bump.default(x,3.);return (x*2,)
def mutate_copy(x):op.bump_copy.default(x,3.);return (x*2,)
def viewed(x):return (op.view.default(x),)
original_run=GraphLowering.run;original_save=GraphLowering.save_output_code
results={'cpu':{},'boundaries':{},'config':{'implicit_fallbacks':False,'registration':'explicit make_fallback in an isolated process'}}
def output_tree(v):
 if isinstance(v,(tuple,list)):return [output_tree(z) for z in v]
 if isinstance(v,dict):return {k:output_tree(z) for k,z in v.items()}
 if isinstance(v,ir.MultiOutput):return {'kind':'MultiOutput','name':v.get_name(),'indices':[[t.__name__,i] for t,i in v.indices],'size':list(map(str,v.get_size())),'stride':list(map(str,v.get_stride()))}
 return {'kind':type(v).__name__,'value':str(v)}
def run(self,*args):
 out=original_run(self,*args);items=[]
 for node in self.operations:
  row={'kind':type(node).__name__,'name':node.get_name()}
  if isinstance(node,ir.FallbackKernel):
   row.update(target=str(node.op_overload),tensor_inputs=[{'name':v.get_name(),'size':list(map(str,v.get_size())),'stride':list(map(str,v.get_stride()))} for v in node.inputs],non_tensor_args=list(map(repr,node.constant_args)),rebuilt=repr(node.unflatten_args([f'T{i}' for i in range(len(node.inputs))],[f'C{i}' for i in range(len(node.constant_args))])),outputs=output_tree(node.outputs),side_effects=node.has_side_effects(),alias_names=node.get_inputs_that_alias_output(),mutation_names=node.get_mutation_names(),mutation_outputs=[{'name':z.get_name(),'mutations':z.get_mutation_names()} for z in node.mutation_outputs],should_allocate=node.should_allocate())
  items.append(row)
 (O/f'{label}_lowered.txt').write_text('\n'.join(repr(n) for n in self.operations)+'\n')
 state['operations']=items;state['mutated_buffers']=list(self.mutated_buffers)
 return out
GraphLowering.run=run
try:
 with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False,'implicit_fallbacks':False}):
  for label,fn,inputs in [('packed',packed,[x,y]),('contiguous',plain,[xs]),('mutation',mutate,[x]),('mutation_copy',mutate_copy,[xs]),('view',viewed,[xs])]:
   state={};tracing_inputs=[v.clone(memory_format=torch.preserve_format) for v in inputs];gm=make_fx(fn)(*tracing_inputs)
   counts.clear();runtime_events.clear();GraphLowering.save_output_code=staticmethod(lambda code:(O/f'{label}_output.py').write_text(code))
   compiled=compile_fx_inner(gm,inputs,is_inference=True)
   compile_counts=dict(counts);assert not any(value for key,value in counts.items() if key.endswith('_cpu'))
   runs=[]
   for iteration in range(2):
    ai=[v.clone(memory_format=torch.preserve_format) for v in inputs];ei=[v.clone(memory_format=torch.preserve_format) for v in inputs]
    before_events=len(runtime_events);actual=compiled(list(ai));actual_events=runtime_events[before_events:].copy();actual_calls=len(actual_events)
    expected=fn(*ei);torch.testing.assert_close(actual,expected)
    assert all(torch.equal(a,e) and a.stride()==e.stride() for a,e in zip(ai,ei))
    aliases=lambda outs,ins:[[o.untyped_storage().data_ptr()==t.untyped_storage().data_ptr() for t in ins] for o in outs]
    assert aliases(actual,ai)==aliases(expected,ei)
    assert [o.stride() for o in actual]==[o.stride() for o in expected]
    assert actual_calls==1
    runs.append({'outputs':[o.tolist() for o in actual],'inputs_after':[a.tolist() for a in ai],'output_input_aliases':aliases(actual,ai),'runtime_events':actual_events,'runtime_calls':actual_calls})
   if label in ['contiguous','mutation_copy']:assert all(r['runtime_events'][0]['stride']==[3,1] for r in runs)
   if label=='view':assert all(r['output_input_aliases']==[[True]] for r in runs)
   results['cpu'][label]={**state,'compile_counts':compile_counts,'input_layouts':[{'shape':list(v.shape),'stride':list(v.stride())} for v in inputs],'runs':runs,'scope':'complete CPU compilation/execution; numeric, input mutation, strides and storage aliases checked'}
  # Mutable custom alias-returning schema is not the special in-place ATen case.
  label='unsupported_mutable_alias';state={};gm=make_fx(lambda x:(op.bad.default(x),))(x.clone())
  try:compile_fx_inner(gm,[x.clone()],is_inference=True)
  except Exception as exc:
   assert 'NYI: Can\'t generate FallbackKernel' in str(exc)
   results['boundaries'][label]={'error':type(exc).__name__,'message':str(exc)[:700]}
  else:raise AssertionError('custom mutable alias case unexpectedly supported')
finally:GraphLowering.run=original_run;GraphLowering.save_output_code=staticmethod(original_save)
# Real fake CUDA output can exist without a CUDA dispatcher kernel.
mode=FakeTensorMode();meta=torch.empty((2,3),device='meta');f=FakeTensor(mode,meta,torch.device('cuda',0))
with mode:fake_output=op.pack.default([f,f],0,alpha=3.,reverse=True)
assert str(fake_output[0].device)=='cuda:0'
keys={key:torch._C._dispatch_has_kernel_for_dispatch_key('inductor_external_lesson::pack',key) for key in ['CPU','CUDA','Meta']}
assert keys=={'CPU':True,'CUDA':False,'Meta':True}
results['boundaries']['fake_is_not_runtime']={'dispatch_keys':keys,'fake_device':str(fake_output[0].device),'scope':'real dispatch registry query and metadata-only FakeTensor CUDA call; no GPU allocation/execution'}
# find_device is a representative device helper, not a transfer or dispatch operation.
first_cpu=types.SimpleNamespace(get_device=lambda:torch.device('cpu'))
selected=ir.FallbackKernel.find_device([first_cpu],fake_output)
assert selected==torch.device('cpu')
assert ir.FallbackKernel.find_device([],fake_output)==torch.device('cuda',0)
assert ir.FallbackKernel.find_device([],{'out':fake_output[0]}) is None
try:ir.FallbackKernel.find_device([],[None,7])
except IndexError:empty_error='IndexError'
else:raise AssertionError('no-device output unexpectedly accepted')
results['boundaries']['find_device']={'first_input_wins':str(selected),'output_sequence_device':'cuda:0','dict_without_inputs':None,'no_tensor_sequence_error':empty_error,'scope':'actual helper with explicit first-input fixture and fake outputs; no device transfer'}
# Registration rejects arbitrary callables and known decomposition conflicts.
for name,target in [('callable',lambda x:x),('decomp',torch.ops.aten.t.default)]:
 try:lowering.make_fallback(target)
 except Exception as exc:results['boundaries'][name]={'error':type(exc).__name__,'message':str(exc)[:250]}
 else:raise AssertionError('invalid fallback registration accepted')
assert results['boundaries']['callable']['error']=='RuntimeError' and results['boundaries']['decomp']['error']=='AssertionError'
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
