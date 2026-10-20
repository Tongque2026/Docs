"""功能化分解的前置决策和清理断言；不执行 Triton。"""
from pathlib import Path
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch._inductor import config
from torch._inductor.virtualized import V
from torch._higher_order_ops.auto_functionalize import auto_functionalized as af,auto_functionalized_v2 as af2
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional as triton_functional
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/functional-wrapper-boundaries';OUT.mkdir(exist_ok=True)
lib=torch.library.Library('inductor_wrapper_boundary_lesson','DEF');lib.define('mut(Tensor(a!) x) -> ()')
def mutate(x):x.add_(1)
lib.impl('mut',mutate,'CPU')
@torch.library.register_fake('inductor_wrapper_boundary_lesson::mut')
def fake_mutate(x):return None
op=torch.ops.inductor_wrapper_boundary_lesson.mut.default
x=torch.arange(4,dtype=torch.float32)
def v1(x):_,new=af(op,x=x);return (new,)
def v2(x):_,new=af2(op,_x_base_index=0,_all_bases=[x]);return (new,)
report={}
# 明确违反前置顺序的受控对照：默认空名单表示不 clone，而不是未知时全 clone。
for name,fn in [('v1_missing_decision',v1),('v2_missing_decision',v2)]:
    gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x]);before=str(gm.graph)
    with V.set_fake_mode(mode):decompose_auto_functionalized(gm.graph)
    gm.graph.lint();gm.recompile();ai=x.clone();ei=x.clone();actual=gm(ai);expected=fn(ei)
    torch.testing.assert_close(actual,expected)
    assert not torch.equal(ai,ei) and torch.equal(ei,x)
    assert actual[0].untyped_storage().data_ptr()==ai.untyped_storage().data_ptr()
    report[name]={'outputs':[o.tolist() for o in actual],'actual_input_after':ai.tolist(),'reference_input_after':ei.tolist(),'same_values_but_changed_mutation':True,'clone_count':sum(n.target==torch.ops.aten.clone.default for n in gm.graph.nodes),'scope':'intentionally skipped reinplace core; consumer precondition experiment'}
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(str(gm.graph)+'\n')
# 前置核心执行后 metadata 明确要求 clone，分解保持外部输入。
for name,fn in [('v1_with_decision',v1),('v2_with_decision',v2)]:
    gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x])
    with V.set_fake_mode(mode):reinplace_inplaceable_ops_core(gm.graph);decompose_auto_functionalized(gm.graph)
    gm.recompile();ai=x.clone();actual=gm(ai);torch.testing.assert_close(actual,fn(x.clone()));assert torch.equal(ai,x)
    report[name]={'input_unchanged':True,'clone_count':sum(n.target==torch.ops.aten.clone.default for n in gm.graph.nodes)}
# 真实 CPU 候选过滤使 handler 不执行，入口末尾必须拒绝仍存在的 wrapper。
for name,fn,message in [('v1_retained',v1,'auto_functionalized was not removed'),('v2_retained',v2,'auto_functionalized_v2 was not removed')]:
    gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x])
    try:
        with config.patch({'disable_cpp_codegen':True}),V.set_fake_mode(mode):decompose_auto_functionalized(gm.graph)
    except AssertionError as e:report[name]={'error':type(e).__name__,'message':str(e),'scope':'real CPU candidate filter and final invariant check'};assert str(e)==message
    else:raise AssertionError('retained node should fail')
# 只建立目标节点和 CPU Fake metadata，验证 Triton 的同一出口检查；无 kernel 执行。
g=torch.fx.Graph();inp=g.placeholder('x');call=g.call_function(triton_functional,(),{'kernel_idx':0,'constant_args_idx':0,'grid':[(1,)],'kwargs':{'X':inp},'tensors_to_clone':['X']});g.output((call,));gm=torch.fx.GraphModule({},g)
from torch._subclasses.fake_tensor import FakeTensorMode
mode=FakeTensorMode();fake=mode.from_tensor(x);inp.meta['val']=fake;call.meta['val']={'X':fake}
try:
    with config.patch({'disable_cpp_codegen':True}),V.set_fake_mode(mode):decompose_auto_functionalized(g)
except AssertionError as e:report['triton_retained']={'error':type(e).__name__,'message':str(e),'scope':'constructed FX target and Fake metadata only; neither kernel lookup nor Triton execution'};assert str(e)=='triton_kernel_wrapper_functional was not removed'
else:raise AssertionError('retained Triton wrapper should fail')
gm=make_fx(lambda x:(x+1,))(x);mode=fake_tensor_prop(gm,[x]);before=str(gm.graph)
with V.set_fake_mode(mode):result=decompose_auto_functionalized(gm.graph)
assert str(gm.graph)==before and result is None
report['no_wrappers']={'graph_unchanged':True,'return_is_none':True}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
