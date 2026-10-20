"""真实局部重排、DCE 与 noop pass；对照数值、输入和 Python 身份。"""
from pathlib import Path
import json,operator
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes.post_grad import reorder_for_locality,remove_noop_ops
from torch._inductor.pattern_matcher import compute_mutation_region_ids,get_mutation_region_id
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/cleanup-noops';OUT.mkdir(exist_ok=True)
report={}
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]]);y=torch.ones_like(x)
clone=lambda args:[v.clone() for v in args]
def pure(x,y):
    a=x.sin();b=y.cos();c=a+1;d=b+2
    return (c,d)
def shared(x,y):
    a=x.sin();c=a+1;b=y.cos();d=a*2
    return (c,b,d)
def before_copy(x,y):
    a=x.sin();x.copy_(y);b=x.cos();c=a+1
    return (c,b)
def other_mutation(x,y):
    a=x.sin();x.add_(1);b=y.cos();c=a+1;x.copy_(y)
    return (c,b,x)
def dead(x,y):
    unused=x.sin();x.copy_(y)
    return (x,)
for name,fn in [('pure',pure),('shared',shared),('copy_tail',before_copy),('other_mutation',other_mutation),('dce',dead)]:
    gm=make_fx(fn)(*clone([x,y]));before=str(gm.graph);before_order=[n.name for n in gm.graph.nodes]
    if name=='dce':gm.graph.eliminate_dead_code()
    else:reorder_for_locality(gm.graph)
    gm.recompile();gm.graph.lint();ai=clone([x,y]);ei=clone([x,y]);actual=gm(*ai);expected=fn(*ei)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(ai,ei):torch.testing.assert_close(a,e)
    regions={n.name:get_mutation_region_id(gm.graph,n) for n in gm.graph.nodes}
    report[name]={'before_order':before_order,'after_order':[n.name for n in gm.graph.nodes],'regions':regions,'outputs':[t.tolist() for t in actual],'inputs_after':[t.tolist() for t in ai],'check':'numeric and input mutation pass'}
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(str(gm.graph)+'\n')

noop_cases={
'clone_internal':lambda x:(x.clone().relu(),),
'clone_output':lambda x:(x.clone(),),
'cat_internal':lambda x:(torch.cat([x]).relu(),),
'cat_output':lambda x:(torch.cat([x]),),
'pow_internal':lambda x:(x.pow(1).relu(),),
'pad_internal':lambda x:(torch.ops.aten.constant_pad_nd.default(x,[0,0]).relu(),),
'cast_changes_dtype':lambda x:(x.to(torch.float64).relu(),),
'primitive_same_dtype':lambda x:(torch.ops.prims.convert_element_type.default(x,torch.float32).relu(),),
'primitive_changes_dtype':lambda x:(torch.ops.prims.convert_element_type.default(x,torch.float64).relu(),),
'alias_tuple':lambda x:(torch.ops.aten.alias.default(x),)}
for name,fn in noop_cases.items():
    gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x]);before=str(gm.graph)
    before_inputs=[str(n.target) for n in gm.graph.nodes if n.op=='call_function']
    raw=gm(x);identity_before=raw[0] is x;storage_before=raw[0].untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
    output=next(iter(reversed(gm.graph.nodes)));node=next((n for n in gm.graph.nodes if n.target==torch.ops.aten.alias.default),None)
    direct_membership=node in output.args if node else None
    remove_noop_ops(gm.graph);gm.recompile();gm.graph.lint();actual=gm(x)
    torch.testing.assert_close(actual,raw)
    report[name]={'before_targets':before_inputs,'after_targets':[str(n.target) for n in gm.graph.nodes if n.op=='call_function'],'identity_before':identity_before,'identity_after':actual[0] is x,'storage_alias_before':storage_before,'storage_alias_after':actual[0].untyped_storage().data_ptr()==x.untyped_storage().data_ptr(),'alias_node_directly_in_output_args':direct_membership,'actual_dtype':str(actual[0].dtype)}
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(str(gm.graph)+'\n')
# 非 tuple 的单输出图：源码的直接 membership 条件在此命中。
gm=make_fx(lambda x:(torch.ops.aten.alias.default(x),))(x);out=next(iter(reversed(gm.graph.nodes)));out.args=(out.args[0][0],);gm.recompile();fake_tensor_prop(gm,[x]);remove_noop_ops(gm.graph);gm.recompile();gm.graph.lint()
report['alias_single_output']={'after_targets':[str(n.target) for n in gm.graph.nodes if n.op=='call_function'],'identity_after':gm(x) is x}
assert torch.ops.aten.alias.default in [n.target for n in gm.graph.nodes]
assert report['alias_tuple']['identity_before'] is False and report['alias_tuple']['identity_after'] is True
assert report['pure']['before_order']!=report['pure']['after_order']
assert 'sin' not in report['dce']['after_order'] and 'copy_' in report['dce']['after_order']
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
