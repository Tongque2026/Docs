"""核心原地化：普通、foreach、功能化包装及 Triton 描述的真实分支。"""
from pathlib import Path
import json,operator,os,contextlib,io
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(Path(__file__).resolve().parent/'cache-reinplace-storage')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes import reinplace as rp
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch._inductor.fx_utils import get_node_storage
from torch._inductor.virtualized import V
from torch._higher_order_ops.auto_functionalize import auto_functionalized as af,auto_functionalized_v2 as af2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/reinplace-storage';OUT.mkdir(exist_ok=True)
a=torch.ops.aten
clone=lambda ins:[x.clone() for x in ins]
x=torch.arange(4,dtype=torch.float32);y=x+10;i=torch.tensor([1]);v=torch.tensor([9.])
def internal(x,i,v):return (a.index_put.default(x+0,[i],v),)
def no_copy(x,i,v):return (a.index_put.default(x,[i],v),)
def copy_back(x,i,v):
    z=a.index_put.default(x,[i],v);x.copy_(z);return (x,)
def old_view(x,i,v):
    z=a.index_put.default(x,[i],v);old=x.view(2,2).sin();x.copy_(z);return (old,x)
def after_copy(x,i,v):
    z=a.index_put.default(x,[i],v);x.copy_(z);return (x+2,x)
def input_view(x,i,v):
    view=x.view(4);return (a.index_put.default(view,[i],v),)
def foreach_full(x,y):
    z=a._foreach_add.Scalar([x,y],1);x.copy_(z[0]);y.copy_(z[1]);return (x,y)
def foreach_partial(x,y):
    z=a._foreach_add.Scalar([x,y],1);x.copy_(z[0]);return (x,z[1])
def foreach_same(x):
    z=a._foreach_add.Scalar([x,x],1);x.copy_(z[0]);x.copy_(z[1]);return (x,)
def targets(gm):return [str(n.target) for n in gm.graph.nodes if n.op=='call_function']
def aliases(out,ins):return [[isinstance(o,torch.Tensor) and o.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in ins] for o in out]
def serial(o):return o.tolist() if isinstance(o,torch.Tensor) else o
def compare(gm,fn,ins):
    ai,ei=clone(ins),clone(ins);actual,expected=gm(*ai),fn(*ei)
    torch.testing.assert_close(actual,expected)
    for aa,ee in zip(ai,ei):torch.testing.assert_close(aa,ee)
    assert aliases(actual,ai)==aliases(expected,ei)
    return {'outputs':[serial(o) for o in actual],'inputs_after':[t.tolist() for t in ai],'output_input_aliases':aliases(actual,ai),'check':'numeric, input mutation and input aliases pass'}
report={}
for name,fn,ins in [('internal',internal,[x,i,v]),('no_copy',no_copy,[x,i,v]),('copy_back',copy_back,[x,i,v]),('old_view',old_view,[x,i,v]),('after_copy',after_copy,[x,i,v]),('input_view',input_view,[x,i,v]),('missing_storage',internal,[x,i,v]),('foreach_full',foreach_full,[x,y]),('foreach_partial',foreach_partial,[x,y]),('foreach_same',foreach_same,[x])]:
    gm=make_fx(fn)(*clone(ins));mode=fake_tensor_prop(gm,clone(ins));before=str(gm.graph)
    if name=='missing_storage':
        next(n for n in gm.graph.nodes if n.target==a.add.Tensor).meta.pop('val')
    storages={n.name:get_node_storage(n) for n in gm.graph.nodes}
    with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(gm.graph)
    gm.graph.lint();gm.recompile();core_text=str(gm.graph);core_targets=targets(gm);boundary={}
    if name=='foreach_full':
        try:
            with contextlib.redirect_stderr(io.StringIO()):gm(*clone(ins))
        except TypeError as e:boundary={'direct_fx_error':type(e).__name__,'reason':'inplace foreach returns None; dead getitem still in core graph'}
        else:raise AssertionError('expected dead getitem eager boundary')
        gm.graph.eliminate_dead_code();gm.recompile()
    report[name]={'targets':core_targets,'after_cleanup_targets':targets(gm),'storage_groups':[[n for n,s in storages.items() if s==key] for key in dict.fromkeys(storages.values()) if key is not None],**boundary,**compare(gm,fn,ins)}
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(core_text+'\n')
    if boundary:(OUT/f'{name}_after_dce.txt').write_text(str(gm.graph)+'\n')
assert 'aten.index_put_.default' in report['internal']['targets']
assert report['copy_back']['targets']==['aten.index_put_.default']
assert all('aten.index_put.default' in report[n]['targets'] for n in ['no_copy','old_view','input_view','missing_storage'])
assert 'aten.index_put_.default' in report['after_copy']['targets']
assert 'aten._foreach_add_.Scalar' in report['foreach_full']['targets']
assert 'aten._foreach_add.Scalar' in report['foreach_partial']['targets'] and 'aten._foreach_add.Scalar' in report['foreach_same']['targets']
# 使用真实图创建静态 shape-only 使用，避免 make_fx 将它折叠成常量。
g=torch.fx.Graph();px=g.placeholder('x');pi=g.placeholder('i');pv=g.placeholder('v');tmp=g.call_function(a.add.Tensor,(px,0));z=g.call_function(a.index_put.default,(tmp,[pi],pv));view=g.call_function(a.view.default,(tmp,[2,2]));size=g.call_function(a.sym_size.int,(view,0));g.output((z,size));gm=torch.fx.GraphModule({},g);mode=fake_tensor_prop(gm,[x,i,v]);before=str(g)
with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(g)
g.lint();gm.recompile();reference=lambda x,i,v:(a.index_put.default(x+0,[i],v),2)
report['metadata_only']={'targets':targets(gm),**compare(gm,reference,[x,i,v])};assert 'aten.index_put_.default' in report['metadata_only']['targets']
(OUT/'metadata_only_before.txt').write_text(before+'\n');(OUT/'metadata_only_after.txt').write_text(str(g)+'\n')
# 完整单图编译沿产品的正常调用位置运行 foreach 原地化。
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor import config
from torch._inductor.graph import GraphLowering
fresh=make_fx(foreach_full)(x.clone(),y.clone());original_save=GraphLowering.save_output_code
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/'foreach_compiled_output.py').write_text(code))
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):compiled=compile_fx_inner(fresh,[x.clone(),y.clone()],is_inference=True)
finally:GraphLowering.save_output_code=staticmethod(original_save)
ai,ei=[x.clone(),y.clone()],[x.clone(),y.clone()];actual=compiled(list(ai));expected=foreach_full(*ei)
torch.testing.assert_close(actual,expected)
for av,ev in zip(ai,ei):torch.testing.assert_close(av,ev)
assert aliases(actual,ai)==aliases(expected,ei)
report['foreach_compiled']={'outputs':[o.tolist() for o in actual],'inputs_after':[v.tolist() for v in ai],'mutated_input_idxs':sorted(compiled.mutated_input_idxs),'output_input_aliases':aliases(actual,ai),'check':'normal full CPU compilation, mutation and alias checks pass'}

# 进程内注册实际 CPU/Fake mutable op；输出由包装携带，不改产品注册。
lib=torch.library.Library('inductor_reinplace_lesson','DEF');lib.define('twice(Tensor(a!) x, Tensor(b!) y) -> ()')
def mutate(x,y):x.add_(1);y.mul_(2)
lib.impl('twice',mutate,'CPU')
@torch.library.register_fake('inductor_reinplace_lesson::twice')
def fake_mutate(x,y):return None
op=torch.ops.inductor_reinplace_lesson.twice.default

def af_internal(x,y):
    u=x+0;v=y+0;_,u,v=af(op,x=u,y=v);return (u,v)
def af_shared(x):
    u=x+0;_,u,v=af(op,x=u,y=u);return (u,v)
def af_old(x,y):
    u=x+0;v=y+0;_,nu,nv=af(op,x=u,y=v);return (nu,nv,u+3)
def af_input(x,y):
    _,u,v=af(op,x=x,y=y);return (u,v)
def af_copy(x,y):
    _,u,v=af(op,x=x,y=y);x.copy_(u);y.copy_(v);return (x,y)
def v2_partial(x,y):
    _,u,v=af2(op,_x_base_index=0,_y_base_index=1,_all_bases=[x+0,y]);return (u,v)
def v2_internal(x,y):
    _,u,v=af2(op,_x_base_index=0,_y_base_index=1,_all_bases=[x+0,y+0]);return (u,v)
def v2_views(x):
    _,out=af2(op,_x_base_index=0,_x_size=[2],_x_stride=[1],_x_storage_offset=0,_y_base_index=0,_y_size=[2],_y_stride=[1],_y_storage_offset=2,_all_bases=[x]);return (out,)
for name,fn,ins in [('af_internal',af_internal,[x,y]),('af_shared',af_shared,[x]),('af_old',af_old,[x,y]),('af_input',af_input,[x,y]),('af_copy',af_copy,[x,y]),('v2_partial',v2_partial,[x,y]),('v2_internal',v2_internal,[x,y]),('v2_views',v2_views,[x])]:
    gm=make_fx(fn)(*clone(ins));mode=fake_tensor_prop(gm,clone(ins));before=str(gm.graph)
    with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(gm.graph)
    decisions=[{'target':str(n.target),'only_clone_these_tensors':n.meta.get('only_clone_these_tensors')} for n in gm.graph.nodes if n.target in (af,af2)]
    core=str(gm.graph)
    # metadata 需由后续分解读取；不能直接执行尚未消费决策的 HOP 图当最终结果。
    with V.set_fake_mode(mode):decompose_auto_functionalized(gm.graph)
    gm.graph.lint();gm.recompile();report[name]={'decisions':decisions,'clone_count':targets(gm).count('aten.clone.default'),'targets':targets(gm),**compare(gm,fn,ins)}
    for stage,text in [('before',before),('core',core),('after',str(gm.graph))]:(OUT/f'{name}_{stage}.txt').write_text(text+'\n')
assert [report[n]['decisions'][0]['only_clone_these_tensors'] for n in ['af_internal','af_shared','af_old','af_input','af_copy','v2_partial','v2_internal','v2_views']]==[[],['y'],['x'],['x','y'],[],[1],[],[0]]
assert [report[n]['clone_count'] for n in ['af_internal','af_shared','af_old','af_input','af_copy','v2_partial','v2_internal','v2_views']]==[0,1,1,2,0,1,0,1]
# FX immutable_list 实现 __hash__；真实列表参数也进入整体可写性判定。
lib.define('list_mut(Tensor(a!)[] xs) -> ()')
def mutate_list(xs):
    for value in xs:value.add_(1)
lib.impl('list_mut',mutate_list,'CPU')
@torch.library.register_fake('inductor_reinplace_lesson::list_mut')
def fake_list(xs):return None
list_op=torch.ops.inductor_reinplace_lesson.list_mut.default
def mutable_list(x,y):
    _,values=af(list_op,xs=[x+0,y+0]);return tuple(values)
def mutable_list_alias(x,y):
    value=x+0;_,values=af(list_op,xs=[value,value]);return tuple(values)
for name,fn,expected_clones in [('mutable_list',mutable_list,[]),('mutable_list_alias',mutable_list_alias,['xs'])]:
    gm=make_fx(fn)(x,y);mode=fake_tensor_prop(gm,[x,y]);hop=next(n for n in gm.graph.nodes if n.target==af)
    container=hop.kwargs['xs'];container_type=type(container).__name__;hash(container)
    with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(gm.graph)
    decisions=list(hop.meta['only_clone_these_tensors'])
    assert decisions==expected_clones
    with V.set_fake_mode(mode):decompose_auto_functionalized(gm.graph)
    gm.graph.lint();gm.recompile()
    report[name]={'container_type':container_type,'only_clone_these_tensors':decisions,'clone_count':targets(gm).count('aten.clone.default'),**compare(gm,fn,[x,y])}
assert report['mutable_list']['clone_count']==0 and report['mutable_list_alias']['clone_count']==2

# Triton 分支仅创建真实 JITFunction 描述、执行核心，不发射 kernel。
try:
    import triton
except ModuleNotFoundError as e:
    report['triton_dependency']={'error':type(e).__name__,'module':e.name,'scope':'Triton descriptor branch not run; package unavailable'}
else:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table,triton_kernel_wrapper_functional
    @triton.jit
    def descriptor_kernel(X):pass
    for name,use_internal,kernel in [('triton_internal',True,descriptor_kernel),('triton_input',False,descriptor_kernel),('triton_unknown',False,object())]:
        gm=make_fx(lambda x:(x+0,))(x);mode=fake_tensor_prop(gm,[x]);g=gm.graph;output=next(n for n in g.nodes if n.op=='output');inp=output.args[0][0] if use_internal else next(n for n in g.nodes if n.op=='placeholder')
        idx=kernel_side_table.add_kernel(kernel)
        with g.inserting_before(output):
            call=g.call_function(triton_kernel_wrapper_functional,(),{'kernel_idx':idx,'constant_args_idx':0,'grid':[(1,)],'kwargs':{'X':inp},'tensors_to_clone':['X']})
            item=g.call_function(operator.getitem,(call,'X'))
        with mode:fake_result=inp.meta['val'].clone()
        call.meta['val']={'X':fake_result};item.meta['val']=fake_result;output.args=((item,),)
        try:
            with V.set_fake_mode(mode):rp.reinplace_inplaceable_ops_core(g)
        except AssertionError as e:
            assert name=='triton_unknown';report[name]={'error':type(e).__name__,'message':str(e),'scope':'descriptor classification only'}
        else:
            g.lint();report[name]={'tensors_to_clone':list(call.kwargs['tensors_to_clone']),'kwargs_type':type(call.kwargs).__name__,'scope':'real JITFunction and graph core only; kernel never executed'}
    assert report['triton_internal']['tensors_to_clone']==[] and report['triton_input']['tensors_to_clone']==['X']
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
