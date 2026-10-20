"""实际规范化、原地化和分解：记录视图路径、存储关系及失败边界。"""
from pathlib import Path
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes import reinplace as rp
from torch._inductor.fx_utils import is_node_realized
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/view-scatter';OUT.mkdir(exist_ok=True)
a=torch.ops.aten
clone=lambda ins:[x.clone() for x in ins]
def _nested(x,s,other):
    row=a.select.int(x,0,1)
    updated=a.slice_scatter.default(row,s,0,1,3)
    whole=a.select_scatter.default(x,updated,0,1)
    return (whole,updated) if other else (whole,)
def nested(x,s):return _nested(x,s,False)
def shared(x,s):return _nested(x,s,True)
def mismatch(x,s):
    row=a.select.int(x,0,1);updated=a.slice_scatter.default(row,s,0,1,3)
    return (a.select_scatter.default(x,updated,0,0),)
def other_base(x,y,s):
    row=a.select.int(y,0,1);updated=a.slice_scatter.default(row,s,0,1,3)
    return (a.select_scatter.default(x,updated,0,1),)
def copy_back(x,s):
    whole=nested(x,s)[0];x.copy_(whole);return (x,)
def internal(x,s):return nested(x+1,s)
def diag(x,s):return (a.diagonal_scatter.default(x,s),)
def strided(x,s):return (a.as_strided_scatter.default(x,s,[2,2],[4,1],5),)
x=torch.arange(12,dtype=torch.float32).reshape(3,4);s=torch.tensor([90.,91.])
cases={'nested':(nested,[x,s]),'shared_inner':(shared,[x,s]),'path_mismatch':(mismatch,[x,s]),'different_base':(other_base,[x,x+20,s]),'copy_back':(copy_back,[x,s]),'internal':(internal,[x,s]),'diagonal':(diag,[x,torch.tensor([90.,91.,92.])]),'as_strided':(strided,[x,torch.tensor([[90.,91.],[92.,93.]])])}
def spec(v):
    if isinstance(v,torch.fx.Node):return {'node':v.name}
    if isinstance(v,(tuple,list)):return [spec(i) for i in v]
    return v
def generalized(gm):
    result=[]
    for n in gm.graph.nodes:
        if n.target not in (rp._generalized_scatter,rp._inplace_generalized_scatter):continue
        result.append({'name':n.name,'target':n.target.__name__,'inp':n.args[0].name,'src':n.args[1].name,'views':[{'target':str(v.target),'args':spec(v.args),'kwargs':dict(v.kwargs)} for v in n.args[2]],'input_realized':is_node_realized(n.args[0]),'output_realized':is_node_realized(n),'should_reinplace':rp.should_reinplace_scatter(n),'always_mutating':rp.scatter_always_uses_mutation(n)})
    return result
def aliases(out,ins):return [[v.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in ins] for v in out]
def state(out):return {'outputs':[v.tolist() for v in out],'output_aliases':aliases(out,out),'offsets':[v.storage_offset() for v in out],'strides':[list(v.stride()) for v in out]}
def execute(gm,fn,ins,interpret=False):
    ai,ei=clone(ins),clone(ins);actual=torch.fx.Interpreter(gm).run(*ai) if interpret else gm(*ai);expected=fn(*ei)
    torch.testing.assert_close(actual,expected)
    for av,ev in zip(ai,ei):torch.testing.assert_close(av,ev)
    assert aliases(actual,ai)==aliases(expected,ei)
    return {**state(actual),'expected_output_aliases':aliases(expected,expected),'input_after':[v.tolist() for v in ai],'output_input_aliases':aliases(actual,ai),'check':'numeric, input mutation and input aliases pass'}
report={}
for name,(fn,ins) in cases.items():
    gm=make_fx(fn)(*clone(ins));mode=fake_tensor_prop(gm,clone(ins));(OUT/f'{name}_before.txt').write_text(str(gm.graph)+'\n')
    with V.set_fake_mode(mode):rp.canonicalize_view_scatter_ops(gm.graph)
    gm.graph.lint();canonical=generalized(gm)
    if name=='nested':
        try:gm.recompile()
        except SyntaxError as e:report['canonical_recompile_boundary']={'error':type(e).__name__,'reason':'ViewOp dataclass repr is not executable generated Python'}
        else:raise AssertionError('expected ViewOp code representation boundary')
    (OUT/f'{name}_canonical.txt').write_text(str(gm.graph)+'\n');canon_result=execute(gm,fn,ins,interpret=True)
    with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(gm.graph)
    core=generalized(gm);(OUT/f'{name}_core.txt').write_text(str(gm.graph)+'\n')
    with V.set_fake_mode(mode):rp.decompose_generalized_scatter(gm.graph)
    gm.graph.lint();gm.recompile();result=execute(gm,fn,ins)
    (OUT/f'{name}_decomposed.txt').write_text(str(gm.graph)+'\n')
    report[name]={'canonical':canonical,'canonical_execution':canon_result,'after_core':core,'final_targets':[str(n.target) for n in gm.graph.nodes if n.op=='call_function'],'final':result}
assert len(report['nested']['canonical'])==1 and len(report['nested']['canonical'][0]['views'])==2
assert len(report['path_mismatch']['canonical'])==len(report['different_base']['canonical'])==2
assert report['nested']['after_core'][0]['target']=='_generalized_scatter'
assert report['copy_back']['after_core'][0]['target']=='_inplace_generalized_scatter'
assert 'aten.copy_.default' in report['copy_back']['final_targets'] and 'aten.clone' not in report['copy_back']['final_targets']
assert report['diagonal']['canonical'][0]['always_mutating'] and 'aten.clone' in report['diagonal']['final_targets']
assert report['as_strided']['canonical'][0]['always_mutating']
assert report['shared_inner']['final']['expected_output_aliases']==[[True,False],[False,True]]
assert report['shared_inner']['final']['output_aliases']==[[True,True],[True,True]]
# 直接 helper 的真实 Tensor 行为，与图构建区别开。
views=[rp.ViewOp(a.select.int,(0,1),{}),rp.ViewOp(a.slice.Tensor,(0,1,3),{})]
for name,target in [('direct_functional',rp._generalized_scatter),('direct_inplace',rp._inplace_generalized_scatter)]:
    inp=x.clone();actual=target(inp,s,views);expected=x.clone();expected[1,1:3]=s
    torch.testing.assert_close(actual,expected)
    report[name]={'output':actual.tolist(),'input_after':inp.tolist(),'same_object':actual is inp,'shares_input':actual.untyped_storage().data_ptr()==inp.untyped_storage().data_ptr()}
    assert (actual is inp)==(target is rp._inplace_generalized_scatter)
inp=x.clone();actual=rp._generalized_scatter(inp,torch.tensor(7.),views)
assert actual[1,1:3].tolist()==[7.,7.]
report['direct_broadcast']={'output':actual.tolist(),'input_unchanged':torch.equal(inp,x),'scope':'generalized helper copy_ broadcasting, not native scatter shape matching'}
for name,src,v in [('copy_error',torch.ones(3),views),('view_error',s,[rp.ViewOp(a.select.int,(0,99),{})])]:
    try:rp._generalized_scatter(x,src,v)
    except (RuntimeError,IndexError) as e:report[name]={'error':type(e).__name__,'message':str(e),'cause':type(e.__cause__).__name__ if e.__cause__ else None}
    else:raise AssertionError(name)
assert report['copy_error']['cause']=='RuntimeError' and report['view_error']['error']=='IndexError'
# 空路径 direct helper 有定义，但 functional 图分解的输入契约要求非空。
g=torch.fx.Graph();inp=g.placeholder('inp');src=g.placeholder('src');node=g.call_function(rp._generalized_scatter,(inp,src,[]));g.output((node,));gm=torch.fx.GraphModule({},g)
mode=fake_tensor_prop(gm,[x,x+1])
try:
    with V.set_fake_mode(mode):rp.decompose_generalized_scatter(g)
except IndexError as e:report['empty_path_decomposition']={'error':type(e).__name__,'scope':'manually constructed generalized node; canonicalizer never creates an empty path'}
else:raise AssertionError('empty path expected to fail functional decomposition')
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{f:v[f] for f in ['canonical','after_core','final_targets','error','same_object','shares_input'] if f in v} for k,v in report.items()},ensure_ascii=False,indent=2))
