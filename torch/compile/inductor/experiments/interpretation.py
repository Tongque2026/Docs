"""只运行真实 Lowering，采集非连续输入、常量和节点环境，不生成 kernel。"""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.graph import GraphLowering
from torch._inductor import ir
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1]
x=torch.arange(6,dtype=torch.float32).reshape(3,2).t()
k=torch.ones(2,3)
s=torch.tensor(2.)
gm=make_fx(lambda x:((x+k)*s,))(x)
mode=fake_tensor_prop(gm,[x])
events=[]
orig=GraphLowering.run_node

def run_node(self,n):
    result=orig(self,n)
    current=result;chain=[]
    while True:
        chain.append(type(current).__name__)
        if isinstance(current,(ir.TensorBox,ir.StorageBox)):current=current.data
        else:break
    entry={'name':n.name,'op':n.op,'target':str(n.target),'ir_chain':chain,'operations_so_far':len(self.operations),'constant_keys':list(self.constants)}
    if isinstance(current,ir.InputBuffer):entry['layout']=str(current.layout)
    events.append(entry)
    return result
GraphLowering.run_node=run_node
try:
    with V.set_fake_mode(mode):
        g=GraphLowering(gm,[x],layout_opt=False,is_inference=True)
        with V.set_graph_handler(g):g.run(x)
finally:GraphLowering.run_node=orig
with V.set_fake_mode(mode),V.set_graph_handler(g):
    lowered='\n'.join(repr(o) for o in g.operations)
report={'input_shape':list(x.shape),'input_stride':list(x.stride()),'events':events,'graph_inputs':list(g.graph_inputs),'constants':list(g.constants),'operations':len(g.operations),'outputs':len(g.graph_outputs),'lowered':lowered,'mode':'real Lowering; no kernel compilation or execution'}
assert report['input_stride']==[1,2]
assert len(g.constants)==1
assert g.graph_inputs['x_1'].get_stride()==[1,2]
(ROOT/'experiments/interpretation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
