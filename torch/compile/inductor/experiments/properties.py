"""受控图改写，验证增量属性传播；不是模型性能实验。"""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_utils import FakeTensorUpdater
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1]
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
before_x=x.clone()
gm=make_fx(lambda x:(torch.relu(x)*2,))(x)
mode=fake_tensor_prop(gm,[x])
nodes=list(gm.graph.nodes)
relu=next(n for n in nodes if n.target==torch.ops.aten.relu.default)
mul=next(n for n in nodes if n.target==torch.ops.aten.mul.Tensor)
shape=lambda n:list(n.meta['val'].shape)
report={'mode':type(mode).__name__,'before':{n.name:shape(n) for n in [relu,mul]},'graph_before':str(gm.graph)}
updater=FakeTensorUpdater(gm.graph)
relu.target=torch.ops.aten.sum.dim_IntList
relu.args=(nodes[0],[1])
report['after_rewrite_before_update']={n.name:shape(n) for n in [relu,mul]}
with V.set_fake_mode(mode):updater.incremental_update()
report['after_update']={n.name:shape(n) for n in [relu,mul]}
assert report['before']==report['after_rewrite_before_update']
assert all(v==[2] for v in report['after_update'].values())
gm.recompile();gm.graph.lint()
actual=gm(x)[0]
torch.testing.assert_close(actual,x.sum(1)*2)
assert torch.equal(x,before_x)
report['graph_after']=str(gm.graph)
report['actual']=actual.tolist()
report['input_unchanged']=True
(ROOT/'experiments/properties.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
