"""真实 CPU 元素代码生成跟踪与明确隔离的公共 codegen 协议。"""
from pathlib import Path
import os,json,types
R=Path(__file__).resolve().parents[1];O=R/'experiments/element-codegen';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'experiments/cache-element-codegen');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch,sympy
from torch._inductor import config,metrics
from torch._inductor.codegen.common import CSE,CSEVariable,KernelArgs,Kernel,OpOverrides
from torch._inductor.codegen.triton import TritonOverrides,TritonCSEVariable
from torch._inductor.loop_body import LoopBody
from torch._inductor.graph import GraphLowering
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.virtualized import V,ops,OpsValue
from torch._inductor.utils import IndentedBuffer
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._ordered_set import OrderedSet as OS
from torch.fx.experimental.proxy_tensor import make_fx
results={'cpu':{},'protocol':{}}
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
xs=torch.arange(6,dtype=torch.float32).reshape(3,2).t();b=torch.tensor([10.,20.,30.])
xf=torch.tensor([[0.1,0.2,0.3],[1.001,2.002,3.003]],dtype=torch.float32)
def shared(x):
 u=torch.exp(x);return (u+1,u*2)
cases=[('shared_exp',shared,[x]),('strided_broadcast',lambda x,b:(x+b,),[xs,b]),('roundtrip_fp16',lambda x:(x.to(torch.float16).to(torch.float32),),[xf]),('fp16_output',lambda x:(x.to(torch.float16),),[xf])]
orig_cse=CSE.generate;orig_body=LoopBody.__call__;orig_save=GraphLowering.save_output_code
try:
 for label,fn,inputs in cases:
  state={'cse':[],'bodies':[]};ids={}
  def kernel_id(k):
   if id(k) not in ids:ids[id(k)]=len(ids)
   return ids[id(k)]
  def generate(self,buffer,expr,*args,**kwargs):
   raw=expr.value if isinstance(expr,OpsValue) else expr
   key=raw.getvalue() if isinstance(raw,IndentedBuffer) else raw
   hit=isinstance(key,str) and key in self.cache
   out=orig_cse(self,buffer,expr,*args,**kwargs)
   if isinstance(V.kernel,Kernel):state['cse'].append({'kernel':kernel_id(V.kernel),'kind':type(V.kernel).__name__,'expr':str(key),'hit_before':hit,'var':str(out),'dtype_at_generate_return':str(getattr(out,'dtype',None))})
   return out
  def body(self,*indices):
   out=orig_body(self,*indices)
   if isinstance(V.kernel,Kernel):
    k=V.kernel;state['bodies'].append({'kernel':kernel_id(k),'kind':type(k).__name__,'indices':str(indices),'indexing_exprs':{k:str(v) for k,v in self.indexing_exprs.items()},'loop_graph':self.root_block.debug_str(),'loads':k.loads.getvalue(),'compute':k.compute.getvalue(),'stores':k.stores.getvalue(),'input_args':dict(k.args.input_buffers),'output_args':dict(k.args.output_buffers)})
   return out
  CSE.generate=generate;LoopBody.__call__=body
  GraphLowering.save_output_code=staticmethod(lambda code:(O/f'{label}_output.py').write_text(code))
  gm=make_fx(fn,decomposition_table=select_decomp_table())(*inputs);state['fx_targets']=[str(n.target) for n in gm.graph.nodes if n.op=='call_function'];metrics.reset()
  with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False,'cpp.simdlen':1}):compiled=compile_fx_inner(gm,inputs,is_inference=True)
  actual_inputs=[t.clone(memory_format=torch.preserve_format) for t in inputs];before=[t.clone() for t in actual_inputs]
  actual=compiled(list(actual_inputs));expected=fn(*inputs)
  if label=='roundtrip_fp16':
   assert torch.equal(actual[0],actual_inputs[0]) and not torch.equal(actual[0],expected[0])
   state['comparison']={'eager_match':False,'equals_fp32_input':True,'max_abs_diff':float((actual[0]-expected[0]).abs().max()),'eager_output':expected[0].tolist(),'scope':'observed scalar CPU fused-cast behavior in this source version; not a successful eager-equivalence check'}
  else:
   torch.testing.assert_close(actual,expected)
   state['comparison']={'eager_match':True}
  assert all(torch.equal(t,v) for t,v in zip(actual_inputs,before))
  assert all(o.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for o in actual for t in actual_inputs)
  state.update(inputs=[{'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'values':t.tolist()} for t in inputs],outputs=[t.tolist() for t in actual],output_strides=[list(t.stride()) for t in actual],input_preserved=True,output_input_aliases=False,generated_kernel_count=metrics.generated_kernel_count,scope='complete CPU compilation/execution; scalar SIMD setting used for readable source, not a performance configuration')
  results['cpu'][label]=state
finally:CSE.generate=orig_cse;LoopBody.__call__=orig_body;GraphLowering.save_output_code=staticmethod(orig_save)
(O/'cpu_snapshot.json').write_text(json.dumps(results['cpu'],ensure_ascii=False,indent=2)+'\n')
exp_events=[e for e in results['cpu']['shared_exp']['cse'] if 'std::exp(' in e['expr']]
assert len(exp_events)==1 and not exp_events[0]['hit_before']
assert results['cpu']['roundtrip_fp16']['comparison']['eager_match'] is False
# Isolated CSE protocol, real implementation with a factory-only current kernel.
factory=types.SimpleNamespace(current_node=None,create_cse_var=lambda name,bounds:CSEVariable(name,bounds))
with V.set_kernel_handler(factory):
 cse=CSE('auto ',';');one=IndentedBuffer();two=IndentedBuffer()
 a=cse.generate(one,'f(x)');again=cse.generate(two,'f(x)',bounds=ValueRanges(0,10));other=cse.generate(two,'g(x)')
 assert a is again and not two.getvalue().count('f(x)') and a.use_count==2
 assert str(a.bounds)=='VR[0, 10]'
 cse.store_cache.update(buf0=a,buf1=other);cse.invalidate(OS([a]))
 assert 'buf0' in cse.store_cache and 'buf1' in cse.invalidated_stores
 clone=cse.clone();assert clone.store_cache is cse.store_cache and clone.varname_map is cse.varname_map
 assert not clone.cache and not clone.reduction_cache
 cloned_var=clone.newvar();assert str(cloned_var)=='tmp2'
 results['protocol']['cse_scope']={'first':one.getvalue(),'second':two.getvalue(),'same_variable':True,'use_count':a.use_count,'bounds':str(a.bounds),'invalidated':list(cse.invalidated_stores),'retained_store':list(cse.store_cache),'clone_shared_store':True,'clone_shared_names':True,'clone_empty_expression_cache':True,'clone_next_name':str(cloned_var),'scope':'real CSE with factory fixture; no target program execution'}
# Actual KernelArgs builds pointer names, size arguments, aliases and byte workspace.
G=types.SimpleNamespace(scheduler=types.SimpleNamespace(mutation_real_name={'old':'x'}),removed_buffers=set(),inplaced_to_remove=set(),wrapper_code=None,get_dtype=lambda name:torch.float32)
with V.set_graph_handler(G),V.set_kernel_handler(types.SimpleNamespace(inplaced_to_remove=set())):
 args=KernelArgs();assert args.input('old')==args.input('x')=='in_ptr0';assert args.output('buf0')=='out_ptr0'
 args.make_inplace('x','buf0');assert args.input('x')==args.output('buf0')=='in_out_ptr0'
 symbol=sympy.Symbol('s0',integer=True);assert args.size(symbol)=='ks0'
 defs,calls,types_=args.cpp_argdefs();aliases=list(args.aliases());assert calls==['buf0','s0']
 results['protocol']['args']={'defs':defs,'calls':calls,'types':types_,'aliases':aliases,'live_outputs':list(args.live_output_buffers()),'scope':'actual KernelArgs + explicit dtype/graph fixture; no tensor allocation'}
 ws=KernelArgs();offset0=ws.workspace(sympy.Integer(16),False);offset1=ws.workspace(sympy.Integer(32),True)
 assert offset0==('ws_ptr',0) and offset1==('ws_ptr',16) and ws.workspace_arg.nbytes==48 and ws.workspace_arg.zero_fill
 py=ws.python_argdefs();assert py[0]==['ws_ptr'] and py[1]==['workspace']
 try:ws.cpp_argdefs()
 except AssertionError as e:workspace_error=str(e)
 else:raise AssertionError('CPU workspace accepted')
 results['protocol']['workspace']={'first':offset0,'second':[offset1[0],int(offset1[1])],'total_bytes':int(ws.workspace_arg.nbytes),'zero_fill':ws.workspace_arg.zero_fill,'cpp_error':workspace_error}
 G.removed_buffers.add('gone')
 try:KernelArgs().input('gone')
 except AssertionError:results['protocol']['removed_arg']={'error':'AssertionError'}
 else:raise AssertionError('removed argument accepted')
# Actual common CSEProxy + an explicitly educational text-only Kernel implementation.
class TextOps(OpOverrides):
 @staticmethod
 def add(x,y):return f'ADD({x}, {y})'
class TextKernel(Kernel):
 overrides=TextOps
 def load(self,name,index):return self.cse.generate(self.loads,f'LOAD({name}, {index})')
 def store(self,name,index,value,mode=None):self.stores.writeline(f'STORE({name}, {index}, {value}, mode={mode})')
 def var_ranges(self):return {}
G=types.SimpleNamespace(scheduler=None,removed_buffers={'dead'},name_to_buffer={})
interpreter=types.SimpleNamespace(current_node=types.SimpleNamespace(target='lesson'))
with V.set_graph_handler(G),V.set_interpreter_handler(interpreter),config.patch({'compute_all_bounds':False}):
 k=TextKernel(increase_kernel_count=False)
 with k:
  value=ops.load('x',sympy.Integer(0));summed=value+value
  ops.store('buf0',sympy.Integer(0),summed)
  forwarded=ops.load('buf0',sympy.Integer(0));assert forwarded.value is summed.value
  ops.store('dead',sympy.Integer(0),summed)
  assert 'dead' in k.cse.store_cache and 'dead' not in k.stores.getvalue()
  k.cse.invalidate(OS());ops.load('buf0',sympy.Integer(0));assert 'buf0' in k.must_keep_buffers
 results['protocol']['proxy']={'loads':k.loads.getvalue(),'compute':k.compute.getvalue(),'stores':k.stores.getvalue(),'must_keep':list(k.must_keep_buffers),'store_names':list(k.store_buffer_names),'forwarded':True,'removed_store_skipped':True,'scope':'real CSEProxy/OpsWrapper + text-only educational Kernel; no device registered or executed'}
# Real Triton expression/mask metadata APIs, no triton import, GPU allocation or launch.
with V.set_kernel_handler(types.SimpleNamespace(min_elem_per_thread=1)):
 bool_expr=TritonOverrides.to_dtype('v',torch.bool,torch.float32)
 uint_expr=TritonOverrides.to_dtype('v',torch.uint8,torch.float32)
 fp8_expr=TritonOverrides.to_dtype('v',torch.float8_e5m2,torch.float32)
 assert V.kernel.min_elem_per_thread==4
 try:TritonOverrides.to_dtype('v',torch.float8_e4m3fn,torch.float8_e5m2)
 except AssertionError:fp8_error='AssertionError'
 else:raise AssertionError('cross-fp8 conversion accepted')
 results['protocol']['triton_dtype']={'bool':bool_expr,'uint8':uint_expr,'fp8':fp8_expr,'min_elem_per_thread':4,'cross_fp8_error':fp8_error,'scope':'actual target expression generation only'}
 mask_a=TritonCSEVariable('tmp_a',ValueRanges.unknown());mask_a.mask_vars.add('xmask')
 mask_b=TritonCSEVariable('tmp_b',ValueRanges.unknown());mask_b.update_on_args('add',(mask_a,sympy.Symbol('r0',integer=True)),{})
 assert set(mask_b.mask_vars)=={'xmask','rmask'}
 results['protocol']['triton_masks']={'masks':sorted(mask_b.mask_vars),'scope':'actual CSE metadata propagation only; no masked memory execution'}
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
