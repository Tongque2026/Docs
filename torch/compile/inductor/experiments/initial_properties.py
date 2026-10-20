"""初次属性建立的六条路径；真实 FakeTensor 传播，无后端 kernel。"""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes.post_grad import view_to_reshape
ROOT=Path(__file__).resolve().parents[1]
x=torch.arange(6,dtype=torch.float32).reshape(2,3);y=torch.ones_like(x)
report={}
def values(gm):return {n.name:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype),'device':str(v.device)} for n in gm.graph.nodes if isinstance((v:=n.meta.get('val')),torch.Tensor)}
gm=make_fx(lambda x:(x.view(6),))(x)
view=next(n for n in gm.graph.nodes if n.op=='call_function')
oldcode=gm.code;oldnode=id(view);view_to_reshape(gm)
mode=fake_tensor_prop(gm,[x])
report['new_mode']={'mode':type(mode).__name__,'allow_non_fake_inputs':mode.allow_non_fake_inputs,'target':str(view.target),'same_node':id(view)==oldnode,'forward_still_old_before_recompile':gm.code==oldcode,'metadata':values(gm)}
assert view.target==torch.ops.aten.reshape.default and id(view)==oldnode and gm.code==oldcode
gm.recompile();assert gm.code!=oldcode
mode=FakeTensorMode(allow_non_fake_inputs=False)
f=mode.from_tensor(x)
gm=make_fx(lambda x:(x.relu(),))(x)
r=fake_tensor_prop(gm,[f])
report['reuse']={'same_mode':r is mode,'allow_after':mode.allow_non_fake_inputs,'metadata':values(gm)}
assert r is mode and not mode.allow_non_fake_inputs
for force in [False,True]:
    gm=make_fx(lambda a,b:(a+b,))(x,y)
    try:r=fake_tensor_prop(gm,[f,y],force_allow_non_fake_inputs=force)
    except AssertionError as e:
        assert not force
        report['mixed_rejected']={'error':type(e).__name__,'message':str(e).splitlines()[0],'allow_after':mode.allow_non_fake_inputs}
    else:
        assert force and r is mode
        report['mixed_allowed']={'same_mode':r is mode,'allow_after':mode.allow_non_fake_inputs,'metadata':values(gm)}
    assert not mode.allow_non_fake_inputs
ambient=FakeTensorMode(allow_non_fake_inputs=False)
gm=make_fx(lambda x:(x+1,))(x)
with torch._guards.tracing(torch._guards.TracingContext(ambient)):
    r=fake_tensor_prop(gm,[x],force_allow_non_fake_inputs=True)
assert r is ambient and not ambient.allow_non_fake_inputs
report['ambient']={'same_mode':r is ambient,'allow_after':ambient.allow_non_fake_inputs,'metadata':values(gm)}
other=FakeTensorMode();g=other.from_tensor(y)
gm=make_fx(lambda a,b:(a+b,))(x,y)
try:fake_tensor_prop(gm,[f,g],force_allow_non_fake_inputs=True)
except AssertionError:report['mode_conflict']={'error':'AssertionError','reason':'two distinct input FakeTensor modes'}
else:raise AssertionError('different fake modes must not be combined')
assert torch.equal(x,torch.arange(6,dtype=torch.float32).reshape(2,3)) and torch.equal(y,torch.ones_like(y))
(ROOT/'experiments/initial_properties.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
