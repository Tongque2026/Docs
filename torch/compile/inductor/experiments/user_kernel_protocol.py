"""真实用户 kernel lowering/IR 的元信息协议；替换 Triton 描述与 mutation 分析，不执行 kernel。"""
from pathlib import Path
import json,types,importlib
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.graph import GraphLowering
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.virtualized import V
from torch._inductor import ir,lowering
R=Path(__file__).resolve().parents[1];O=R/'experiments/template-integration';O.mkdir(exist_ok=True)
x=torch.arange(6,dtype=torch.float32);out=torch.zeros_like(x)
gm=make_fx(lambda x,y:(x,y))(x,out);mode=fake_tensor_prop(gm,[x,out]);tw=importlib.import_module('torch._higher_order_ops.triton_kernel_wrap')
orig_get=ir.UserDefinedTritonKernel.get_kernel_and_configs;orig_mut=tw.identify_mutated_tensors
kernel=types.SimpleNamespace(arg_names=['inp','out','BLOCK','scale'],constexprs=[2,3]);configs=[types.SimpleNamespace(kwargs={'BLOCK':64})]
analysis=[]
def identify(kernel,kwargs):analysis.append(list(kwargs));return ['out']
class Wrapper:
 def __init__(self):self.events=[]
 def define_user_defined_triton_kernel(self,k,cs,kwargs):self.events.append({'define':list(kwargs)});return 'lesson_kernel',{'signature':'fixture'}
 def generate_user_defined_triton_kernel(self,name,args,grid,cs,meta,indices):self.events.append({'call':name,'args':[a.get_name() if isinstance(a,ir.IRNode) else a for a in args],'grid':grid,'constexpr_indices':indices})
try:
 ir.UserDefinedTritonKernel.get_kernel_and_configs=lambda self:(kernel,configs);tw.identify_mutated_tensors=identify
 with V.set_fake_mode(mode):
  g=GraphLowering(gm,[x,out],layout_opt=False)
  with V.set_graph_handler(g):
   g.run(x,out);args=list(g.graph_inputs.values());constant_idx=tw.kernel_side_table.add_constant_args({'scale':2.0})
   result=lowering.triton_kernel_wrap_(kernel_idx=7,constant_args_idx=constant_idx,grid=[(1,1,1)],kwargs={'inp':args[0],'out':args[1],'scale':1.0})
   node=g.operations[-1];assert isinstance(node,ir.UserDefinedTritonKernel)
   assert result['inp'] is args[0] and result['out'] is args[1] and set(result)=={'inp','out'}
   assert node.kwargs['scale']==2.0 and node.ordered_kwargs_for_cpp_kernel==['inp','out','scale']
   wrapper=Wrapper();node.codegen_comment=lambda wrapper:None;node.codegen(wrapper)
   assert wrapper.events[-1]['constexpr_indices']==[2] and wrapper.events[-1]['args'][-1]==2.0
   assert list(g.mutated_buffers)==['y_1'] and len(node.mutation_outputs)==1
   try:ir.UserDefinedTritonKernel(kernel_idx=7,grid=[(1,1,1)],kernel_args={'scale':2.0})
   except AssertionError:empty='AssertionError'
   else:raise AssertionError('tensor-free user kernel accepted')
   report={'return_keys':list(result),'same_tensorboxes':True,'ordered_args':node.ordered_kwargs_for_cpp_kernel,'constant_overrode_scale':node.kwargs['scale'],'analysis_kwargs':analysis,'mutated_buffers':list(g.mutated_buffers),'mutation_outputs':[b.get_mutation_names() for b in node.mutation_outputs],'wrapper_events':wrapper.events,'no_tensor_error':empty,'scope':'actual lowering and UserDefinedTritonKernel constructor/codegen with descriptor, identify_mutated_tensors and wrapper fixtures; metadata comment skipped; CPU IR operands; no Triton import/compilation/launch or new device support'}
finally:ir.UserDefinedTritonKernel.get_kernel_and_configs=orig_get;tw.identify_mutated_tensors=orig_mut
(O/'user-kernel-protocol.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
