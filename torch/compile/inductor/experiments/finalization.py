"""真实排序运行 + 假设备元数据上的 constructor pass；不执行 GPU。"""
from pathlib import Path
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensor,FakeTensorMode
from torch._inductor.pattern_matcher import stable_topological_sort
from torch._inductor.fx_passes.post_grad import ConstructorMoverPass
ROOT=Path(__file__).resolve().parents[1]
report={}
x=torch.arange(4,dtype=torch.float32)
gm=make_fx(lambda x:((x+1)*2,))(x)
a=next(n for n in gm.graph.nodes if n.target==torch.ops.aten.add.Tensor)
b=next(n for n in gm.graph.nodes if n.target==torch.ops.aten.mul.Tensor)
a.prepend(b)
report['sort']={'before':[n.name for n in gm.graph.nodes]}
stable_topological_sort(gm.graph)
report['sort']['after']=[n.name for n in gm.graph.nodes]
first=list(gm.graph.nodes);stable_topological_sort(gm.graph)
assert first==list(gm.graph.nodes)
gm.recompile();gm.graph.lint()
torch.testing.assert_close(gm(x)[0],(x+1)*2)
report['sort']['actual']=gm(x)[0].tolist()
a.args=(b,1)
try:stable_topological_sort(gm.graph)
except AssertionError:report['cycle']='AssertionError'
else:raise AssertionError('cyclic graph must be rejected')
mode=FakeTensorMode()
def val(shape,device,dtype=torch.float32):return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype),torch.device(device))
def build(case):
    g=torch.fx.Graph()
    p=g.placeholder('p');p.meta['val']=val([2],'cuda:0')
    if case=='multiple_devices':
        q=g.placeholder('q');q.meta['val']=val([2],'cuda:1')
    shape=[] if case=='boolean_index' else [2]
    c=g.call_function(torch.ops.aten.full.default,(shape,3.),{'dtype':torch.float32,'device':torch.device('cpu')});c.meta['val']=val(shape,'cpu')
    if case=='boolean_index':
        mask=g.placeholder('mask');mask.meta['val']=val([2],'cuda:0',torch.bool)
        z=g.call_function(torch.ops.aten.index_put.default,(p,[mask],c))
    else:z=g.call_function(torch.ops.aten.copy.default,(p,c))
    z.meta['val']=val([2],'cuda:0')
    g.output((c,z) if case=='output_cpu' else (z,))
    m=torch.fx.GraphModule({},g)
    before=str(c.kwargs['device'])
    ConstructorMoverPass('cuda')(g)
    g.lint()
    return {'constructor_before':before,'constructor_after':str(c.kwargs['device']),'graph':str(g),'execution':'metadata-only; no GPU kernels'}
for case in ['movable','output_cpu','multiple_devices','boolean_index']:report[case]=build(case)
assert report['movable']['constructor_after']=='cuda:0'
assert all(report[n]['constructor_after']=='cpu' for n in ['output_cpu','multiple_devices','boolean_index'])
(ROOT/'experiments/finalization.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
