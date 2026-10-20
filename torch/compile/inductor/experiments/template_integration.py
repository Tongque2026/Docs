"""专用 lowering 接入：真实 CPU 外部/模板/复合路径；GPU 只核对元信息与接口。"""
from pathlib import Path
import os,json,types
R=Path(__file__).resolve().parents[1];O=R/'experiments/template-integration';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'experiments/cache-template-integration');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch,sympy
from torch._inductor import config,ir,lowering
from torch._inductor.graph import GraphLowering
from torch._inductor.compile_fx import compile_fx_inner,fake_tensor_prop
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import BackendFeature
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet as OS
A=torch.arange(6,dtype=torch.float32).reshape(2,3);B=torch.arange(12,dtype=torch.float32).reshape(3,4)/8
W=torch.arange(16*16,dtype=torch.float32).reshape(16,16)/256
X=torch.arange(8*16,dtype=torch.float32).reshape(8,16)/128
W_original=W.clone()
BA=torch.arange(6,dtype=torch.float32).reshape(2,1,3);BB=torch.arange(24,dtype=torch.float32).reshape(2,3,4)/8
BA2=torch.arange(12,dtype=torch.float32).reshape(2,2,3)
def constant_mm(x):return (x@W,)
def foreach(x,y):return tuple(torch._foreach_add([x,y],2.))
results={'cpu':{},'boundaries':{},'lowering_only':{}}
orig_select=AlgorithmSelectorCache.__call__;orig_bench=AlgorithmSelectorCache.make_benchmark_fn;orig_run=GraphLowering.run;orig_save=GraphLowering.save_output_code
state={};label=''
def selector(self,name,choices,input_nodes,layout,*args,**kwargs):
 row={'name':name,'choices':[{'kind':type(c).__name__,'name':c.name,'wrapped_kind':type(getattr(c,'_wrapped',None)).__name__} for c in choices if c is not None],'inputs':[{'size':list(map(str,v.get_size())),'stride':list(map(str,v.get_stride())),'device':str(v.get_device())} for v in input_nodes],'layout':{'kind':type(layout).__name__,'size':list(map(str,layout.size)),'stride':list(map(str,layout.stride))},'return_multi_template_requested':kwargs.get('return_multi_template')}
 state.setdefault('selectors',[]).append(row);out=orig_select(self,name,choices,input_nodes,layout,*args,**kwargs);row['result_kind']=type(out.data.data).__name__ if isinstance(out,ir.TensorBox) else type(out).__name__;return out
def benchmark(self,*args,**kwargs):state['benchmark_fn_calls']=state.get('benchmark_fn_calls',0)+1;return orig_bench(self,*args,**kwargs)
def run(self,*args):
 out=orig_run(self,*args);state['operations']=[{'kind':type(n).__name__,'name':n.get_name()} for n in self.operations];state['lists']=dict(self.lists);state['constants']={k:{'shape':list(v.shape),'stride':list(v.stride())} for k,v in self.constants.items()};state['mutated_inputs']=list(self.mutated_inputs);(O/f'{label}_lowered.txt').write_text('\n'.join(repr(n) for n in self.operations)+'\n');return out
AlgorithmSelectorCache.__call__=selector;AlgorithmSelectorCache.make_benchmark_fn=benchmark;GraphLowering.run=run
cases=[('mm_external',lambda a,b:(torch.relu(a@b),),[A,B],{}),('mm_cpp',constant_mm,[X],{'max_autotune_gemm':True,'max_autotune_gemm_backends':'CPP','autotune_fallback_to_aten':False}),('mm_empty_fallback',lambda a,b:(a@b,),[A,B],{'max_autotune_gemm':True,'max_autotune_gemm_backends':'TRITON','autotune_fallback_to_aten':True}),('bmm_empty_backends',lambda a,b:(torch.bmm(a,b),),[BA2,BB],{'max_autotune_gemm':True,'max_autotune_gemm_backends':'TRITON','autotune_fallback_to_aten':False}),('bmm_decomposed',lambda a,b:(torch.bmm(a,b),),[BA,BB],{}),('bmm_external',lambda a,b:(torch.bmm(a,b),),[BA2,BB],{}),('foreach_cpu',foreach,[A,A+1],{})]
try:
 with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False,'max_autotune':False,'max_autotune_gemm':False,'cpp.threads':1}):
  for label,fn,inputs,patch in cases:
   state={'benchmark_fn_calls':0,'selectors':[]};expected=fn(*inputs);gm=make_fx(fn)(*inputs);GraphLowering.save_output_code=staticmethod(lambda code:(O/f'{label}_output.py').write_text(code))
   with config.patch(patch):compiled=compile_fx_inner(gm,inputs,is_inference=True)
   ai=[v.clone() for v in inputs];before=[v.clone() for v in ai];actual=compiled(list(ai));assert torch.equal(W,W_original)
   torch.testing.assert_close(actual,expected);assert all(torch.equal(a,b) for a,b in zip(ai,before))
   assert all(o.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for o in actual for v in ai)
   state.update(captured_weight_preserved=True,output_shapes=[list(v.shape) for v in actual],output_values=[v.tolist() for v in actual],input_preserved=True,output_aliases_input=False,config=patch,scope='full CPU compilation/execution; actual choices and IR captured, no fabricated timings')
   results['cpu'][label]=state
  label='mm_empty_rejected';state={'benchmark_fn_calls':0,'selectors':[]};gm=make_fx(lambda a,b:(a@b,))(A,B)
  try:
   with config.patch({'max_autotune_gemm':True,'max_autotune_gemm_backends':'TRITON','autotune_fallback_to_aten':False}):compile_fx_inner(gm,[A,B],is_inference=True)
  except Exception as exc:
   assert 'No choices to select' in str(exc);results['boundaries'][label]={**state,'error':type(exc).__name__,'message':str(exc)[:600]}
  else:raise AssertionError('empty candidate set accepted')
finally:
 AlgorithmSelectorCache.__call__=orig_select;AlgorithmSelectorCache.make_benchmark_fn=orig_bench;GraphLowering.run=orig_run;GraphLowering.save_output_code=staticmethod(orig_save)
assert any(n['kind']=='CppTemplateBuffer' for n in results['cpu']['mm_cpp']['operations'])
assert all(v['benchmark_fn_calls']==0 for v in results['cpu'].values())
assert not results['cpu']['bmm_decomposed']['selectors'] and not results['cpu']['mm_empty_fallback']['selectors']
assert results['cpu']['foreach_cpu']['lists']=={}
# Only Lowering: explicit FOREACH feature provider to observe grouping, no CPU backend support claimed.
orig_has=GraphLowering.has_feature
try:
 GraphLowering.has_feature=lambda self,device,feature:True if feature is BackendFeature.FOREACH else orig_has(self,device,feature)
 gm=make_fx(foreach)(A,A+1);mode=fake_tensor_prop(gm,[A,A+1])
 with V.set_fake_mode(mode):
  g=GraphLowering(gm,[A,A+1],layout_opt=False)
  with V.set_graph_handler(g):g.run(A,A+1)
 assert len(g.lists)==1
 results['lowering_only']['foreach_feature']={'lists':dict(g.lists),'operation_types':[type(n).__name__ for n in g.operations],'scope':'real Lowering with explicit capability provider override; no Scheduler/codegen/runtime, not CPU native foreach support'}
finally:GraphLowering.has_feature=orig_has
# Actual device gates, with only the hardware-size and feature queries replaced by explicit fixtures.
import torch._inductor.utils as utils
import torch._inductor.codegen.common as common
orig_big=utils.is_big_gpu;orig_feature=common.has_backend_feature
try:
 utils.is_big_gpu=lambda index:True;common.has_backend_feature=lambda dev,feature:True
 with config.patch({'max_autotune_gemm':True,'max_autotune_gemm_backends':'TRITON'}):
  gates={dev:utils.use_triton_template(ir.FixedLayout(torch.device(dev),torch.float32,[2,4])) for dev in ['cpu','cuda:0','xpu:0','privateuseone:0']}
  disabled_dtype=utils.use_triton_template(ir.FixedLayout(torch.device('cuda',0),torch.int64,[2,4]))
 assert gates=={'cpu':False,'cuda:0':True,'xpu:0':False,'privateuseone:0':False} and not disabled_dtype
 results['boundaries']['triton_device_gate']={'gates':gates,'int64':disabled_dtype,'scope':'actual predicate with hardware-size and backend-feature providers explicitly fixed True; no GPU runtime queried'}
finally:utils.is_big_gpu=orig_big;common.has_backend_feature=orig_feature
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
