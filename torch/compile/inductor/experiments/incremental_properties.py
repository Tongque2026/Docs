"""运行真实增量维护，记录结构、属性、存储和停止传播的区别。"""
import json
from pathlib import Path
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_utils import FakeTensorUpdater,get_node_storage
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1]
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
report={}
def build(fn):
    gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x]);return gm,mode
# 结构变成 sum，使用者虽然结构未改，也应更新 shape。
gm,mode=build(lambda x:(x.relu()*2,));nodes=list(gm.graph.nodes)
a=next(n for n in nodes if n.target==torch.ops.aten.relu.default);b=next(n for n in nodes if n.target==torch.ops.aten.mul.Tensor)
u=FakeTensorUpdater(gm.graph);a.target=torch.ops.aten.sum.dim_IntList;a.args=(nodes[0],[1])
with V.set_fake_mode(mode):u.incremental_update()
assert tuple(a.meta['val'].shape)==tuple(b.meta['val'].shape)==(2,)
report['shape_change']={'a_shape':list(a.meta['val'].shape),'b_shape':list(b.meta['val'].shape),'a_signature_saved':u.hash_node(a) in u.processed_hashes,'b_signature_saved':u.hash_node(b) in u.processed_hashes}
# 结构变成 neg，但仅属性相等：新签名未登记，下次调用会再次执行。
gm,mode=build(lambda x:(x.relu(),));a=next(n for n in gm.graph.nodes if n.target==torch.ops.aten.relu.default)
u=FakeTensorUpdater(gm.graph);old=a.meta['val'];a.target=torch.ops.aten.neg.default
count=[0];original_call=torch._ops.OpOverload.__call__
def observed(op,*args,**kwargs):
    if op is torch.ops.aten.neg.default:count[0]+=1
    return original_call(op,*args,**kwargs)
torch._ops.OpOverload.__call__=observed
try:
    with V.set_fake_mode(mode):u.incremental_update()
    first=count[0]
    with V.set_fake_mode(mode):u.incremental_update()
finally:torch._ops.OpOverload.__call__=original_call
report['equal_attributes']={'same_metadata_object':a.meta['val'] is old,'new_signature_saved':u.hash_node(a) in u.processed_hashes,'neg_calls_first':first,'neg_calls_total':count[0]}
assert a.meta['val'] is old and u.hash_node(a) not in u.processed_hashes and count[0]>first>0
# 缺失上游属性，变化节点被跳过。
gm,mode=build(lambda x:(x.relu(),));nodes=list(gm.graph.nodes);a=nodes[1];u=FakeTensorUpdater(gm.graph);old=a.meta['val'];del nodes[0].meta['val'];a.target=torch.ops.aten.neg.default
with V.set_fake_mode(mode):u.incremental_update()
report['missing_input_metadata']={'same_metadata_object':a.meta['val'] is old,'signature_saved':u.hash_node(a) in u.processed_hashes}
assert a.meta['val'] is old and u.hash_node(a) not in u.processed_hashes
# dtype-only 变化：核对当前比较器实际未显式比较 dtype 的行为。
gm,mode=build(lambda x:(x.relu(),));nodes=list(gm.graph.nodes);a=nodes[1];u=FakeTensorUpdater(gm.graph)
a.target=torch.ops.aten._to_copy.default;a.args=(nodes[0],);a.kwargs={'dtype':torch.float64}
with V.set_fake_mode(mode):u.incremental_update()
metadata_dtype=str(a.meta['val'].dtype);gm.recompile();actual=gm(x)[0]
report['dtype_only']={'metadata_dtype':metadata_dtype,'actual_dtype':str(actual.dtype),'new_signature_saved':u.hash_node(a) in u.processed_hashes,'note':'observed checkout behavior, not a product-source fix'}
assert metadata_dtype=='torch.float32' and actual.dtype==torch.float64
# 同 shape 的旧 view 别名变为 clone，存储关系变化应传播。
gm,mode=build(lambda x:(torch.ops.aten.alias.default(x)*2,));nodes=list(gm.graph.nodes);a=next(n for n in nodes if n.target==torch.ops.aten.alias.default);u=FakeTensorUpdater(gm.graph)
before_alias=get_node_storage(a)==get_node_storage(nodes[0]);a.target=torch.ops.aten.clone.default
with V.set_fake_mode(mode):u.incremental_update()
after_alias=get_node_storage(a)==get_node_storage(nodes[0])
report['alias_to_clone']={'before_aliases_input':before_alias,'after_aliases_input':after_alias,'signature_saved':u.hash_node(a) in u.processed_hashes}
assert before_alias and not after_alias
(ROOT/'experiments/incremental_properties.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
