"""直接运行真实图 pass；比较数值、输入修改与输出别名。"""
from pathlib import Path
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes.post_grad import post_grad_passes,view_to_reshape
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/rewrites-mutation';OUT.mkdir(exist_ok=True)
clone=lambda inputs:[x.clone() for x in inputs]
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
y=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
b=torch.tensor([10.,20.])

def internal(x,i,v): return (torch.index_put(x+0,[i],v),)
def input_only(x,i,v): return (torch.index_put(x,[i],v),)
def copy_back(x,i,v):
    z=torch.index_put(x,[i],v)
    x.copy_(z)
    return (x,)
def old_reader(x,i,v):
    z=torch.index_put(x,[i],v)
    old=x+2
    x.copy_(z)
    return (old,x)

cases={
'addmm_tensor':('post',lambda a,b,c:(a@b+c,),[x,y,b]),
'addmm_scalar':('post',lambda a,b,c:(a@b+1,),[x,y,b]),
'internal':('mutation',internal,[torch.arange(4,dtype=torch.float32),torch.tensor([1]),torch.tensor([9.])]),
'input_only':('mutation',input_only,[torch.arange(4,dtype=torch.float32),torch.tensor([1]),torch.tensor([9.])]),
'copy_back':('mutation',copy_back,[torch.arange(4,dtype=torch.float32),torch.tensor([1]),torch.tensor([9.])]),
'old_reader':('mutation',old_reader,[torch.arange(4,dtype=torch.float32),torch.tensor([1]),torch.tensor([9.])]),
}
report={}
for name,(kind,fn,inputs) in cases.items():
    gm=make_fx(fn)(*clone(inputs))
    view_to_reshape(gm)
    mode=fake_tensor_prop(gm,clone(inputs))
    before=str(gm.graph)
    with V.set_fake_mode(mode):
        if kind=='post':post_grad_passes(gm,is_inference=True)
        else:reinplace_inplaceable_ops(gm.graph)
    gm.recompile();gm.graph.lint()
    after=str(gm.graph)
    expected_inputs=clone(inputs);actual_inputs=clone(inputs)
    expected=fn(*expected_inputs);actual=gm(*actual_inputs)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(actual_inputs,expected_inputs):torch.testing.assert_close(a,e)
    alias=lambda t,inp:t.untyped_storage().data_ptr()==inp.untyped_storage().data_ptr()
    assert [alias(t,actual_inputs[0]) for t in actual]==[alias(t,expected_inputs[0]) for t in expected]
    (OUT/f'{name}_before.txt').write_text(before+'\n')
    (OUT/f'{name}_after.txt').write_text(after+'\n')
    report[name]={'before_targets':[str(n.target) for n in make_fx(fn)(*clone(inputs)).graph.nodes if n.op=='call_function'], 'after_targets':[str(n.target) for n in gm.graph.nodes if n.op=='call_function'], 'output':[t.tolist() for t in actual], 'input0_after':actual_inputs[0].tolist(), 'output_alias_input0':[alias(t,actual_inputs[0]) for t in actual], 'numeric_mutation_alias_checks':'pass'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
