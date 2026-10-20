"""真实 matcher/entry 实验：隔离注册，不改全局产品规则。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/pattern-protocol';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-pattern-protocol')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import pattern_matcher as pm,config
from torch._inductor.compile_fx import fake_tensor_prop,compile_fx_inner
from torch._inductor.virtualized import V
from torch._inductor.lowering import lowerings
from torch._inductor.graph import GraphLowering
aten=torch.ops.aten
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
clone=lambda a:[v.clone() for v in a]
def double(x):return (aten.neg.default(aten.neg.default(x)),)
def many(x):
    z=aten.neg.default(x)
    return (aten.neg.default(z),aten.sin.default(z))
def root_many(x):
    z=aten.neg.default(aten.neg.default(x))
    return (z,z+1)
def barrier(x):
    z=aten.neg.default(x)
    x.add_(1)
    return (aten.neg.default(z),x)
def pattern():return pm.CallFunction(aten.neg.default,pm.CallFunction(aten.neg.default,pm.KeywordArg('x')))
def summary(m):
    return {'nodes':[n.name for n in m.nodes],'args':[str(a) for a in m.args], 'kwargs':{k:str(v) for k,v in m.kwargs.items()},'output_nodes':[n.name if n is not None else None for n in m.output_nodes()]}
def targets(gm):return [str(n.target) for n in gm.graph.nodes if n.op=='call_function']
def check_result(gm,fn,inputs):
    ai,ei=clone(inputs),clone(inputs)
    actual,expected=gm(*ai),fn(*ei)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(ai,ei):torch.testing.assert_close(a,e)
    aliases=lambda out,ins:[[o.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in ins] for o in out]
    assert aliases(actual,ai)==aliases(expected,ei)
    return {'outputs':[o.tolist() for o in actual],'inputs_after':[v.tolist() for v in ai],'output_input_aliases':aliases(actual,ai),'checks':'numeric, input mutation, storage alias pass'}
report={}
for name,fn,allowed in [('graph_success',double,True),('graph_extra_false',double,False),('inner_multiuser',many,True),('root_multiuser',root_many,True),('mutation_barrier',barrier,True)]:
    gm=make_fx(fn)(*clone([x]));mode=fake_tensor_prop(gm,clone([x]));p=pm.PatternMatcherPass();events=[]
    def extra(m):events.append({'phase':'extra_check','match':summary(m)});return allowed
    @pm.register_graph_pattern(pattern(),extra_check=extra,pass_dict=p)
    def handler(m,x):
        events.append({'phase':'graph_handler','arg_type':type(x).__name__})
        new=m.graph.call_function(aten.mul.Tensor,(x,1))
        new.meta.update(m.output_node().meta)
        m.output_node().replace_all_uses_with(new);m.erase_nodes()
    before=str(gm.graph)
    with V.set_fake_mode(mode):count=p.apply(gm)
    gm.graph.lint();gm.recompile()
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(str(gm.graph)+'\n')
    report[name]={'count':count,'events':events,'targets':targets(gm),'regions':{n.name:n.meta.get('mutation_region_id') for n in gm.graph.nodes},**check_result(gm,fn,[x])}
assert [report[k]['count'] for k in ['graph_success','graph_extra_false','inner_multiuser','root_multiuser','mutation_barrier']]==[1,0,0,1,0]
assert report['inner_multiuser']['events']==report['mutation_barrier']['events']==[]
# 相同命名捕获必须指向同一节点，不能只凭 shape 相同。
for name,fn in [('named_same',lambda x,y:(x+x,)),('named_different',lambda x,y:(x+y,))]:
    gm=make_fx(fn)(x,x+1)
    n=next(n for n in gm.graph.nodes if n.op=='call_function')
    m=pm.CallFunction(aten.add.Tensor,pm.KeywordArg('x'),pm.KeywordArg('x')).match(n)
    report[name]={'matched':bool(m),'detail':summary(m) if m else str(m)}
assert report['named_same']['matched'] and not report['named_different']['matched']
# apply 计数不等于修改次数：同一根可以继续尝试后续 entry。
p=pm.PatternMatcherPass();events=[]
@pm.register_graph_pattern(pattern(),pass_dict=p)
def observe(m,x):events.append('observe_without_edit')
@pm.register_graph_pattern(pattern(),pass_dict=p)
def edit(m,x):
    events.append('edit');n=m.graph.call_function(aten.mul.Tensor,(x,1));n.meta.update(m.output_node().meta)
    m.output_node().replace_all_uses_with(n);m.erase_nodes()
gm=make_fx(double)(x);mode=fake_tensor_prop(gm,[x])
with V.set_fake_mode(mode):count=p.apply(gm)
gm.graph.lint();gm.recompile();report['same_root_entries']={'count':count,'events':events,**check_result(gm,double,[x])}
assert count==2 and events==['observe_without_edit','edit']
# 新根不会加入 apply 开始时生成的候选快照。
p=pm.PatternMatcherPass();events=[]
@pm.register_graph_pattern(pm.CallFunction(aten.neg.default,pm.KeywordArg('x')),pass_dict=p)
def same_target(m,x):
    events.append(m.output_node().name);n=m.graph.call_function(aten.neg.default,(x,));n.meta.update(m.output_node().meta)
    m.output_node().replace_all_uses_with(n);m.erase_nodes()
fn=lambda x:(-x,)
gm=make_fx(fn)(x);mode=fake_tensor_prop(gm,[x])
with V.set_fake_mode(mode):count=p.apply(gm)
gm.graph.lint();gm.recompile();report['candidate_snapshot']={'count':count,'events':events,'targets':targets(gm),**check_result(gm,fn,[x])}
assert count==1 and len(events)==1
# 注册函数式 replacement：真实重追踪搜索和替代函数。
p=pm.PatternMatcherPass();events=[]
def search(x):return aten.neg.default(aten.neg.default(x))
def replacement(x):return aten.mul.Tensor(x,1)
def trace(fn,args):
    events.append({'phase':'trace','fn':fn.__name__,'shape':[list(a.shape) for a in args]})
    return pm.fwd_only(fn,args)
def extra(m):events.append({'phase':'specific_extra','match':summary(m)});return True
pm.register_replacement(search,replacement,[x],trace,p,extra_check=extra)
registration_events=list(events);events.clear()
gm=make_fx(double)(x);mode=fake_tensor_prop(gm,[x]);before=str(gm.graph)
with V.set_fake_mode(mode):count=p.apply(gm)
gm.graph.lint();gm.recompile()
(OUT/'replacement_before.txt').write_text(before+'\n');(OUT/'replacement_after.txt').write_text(str(gm.graph)+'\n')
report['replacement_entry']={'count':count,'registration_events':registration_events,'application_events':events,'targets':targets(gm),**check_result(gm,double,[x])}
assert count==1 and [e['phase'] for e in events]==['trace','specific_extra','trace']
# lowering handler 接收 IR，而不是在 apply 时执行。
p=pm.PatternMatcherPass();events=[]
@pm.register_lowering_pattern(pattern(),pass_dict=p)
def lower_double(m,x):
    events.append({'arg_type':type(x).__name__,'match_nodes':[n.name for n in m.nodes]})
    return lowerings[aten.mul.Tensor](x,1)
gm=make_fx(double)(x);mode=fake_tensor_prop(gm,[x])
with V.set_fake_mode(mode):count=p.apply(gm)
gm.graph.lint();gm.recompile();(OUT/'lowering_after.txt').write_text(str(gm.graph)+'\n')
after_apply=len(events)
try:
    fake_tensor_prop(gm,[x.clone()])
except StopIteration as e:
    report['lowered_graph_repropagation']={'error':type(e).__name__,'events':list(events),'meaning':'IR handler received FakeTensor before GraphLowering'}
else:
    raise AssertionError('expected IR-only handler to reject FakeTensor')
assert report['lowered_graph_repropagation']['events'][0]['arg_type']=='FakeTensor'
events.clear()
orig_save=GraphLowering.save_output_code
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/'lowering_output.py').write_text(code))
try:
    def install(graph):
        installed=p.apply(graph)
        assert installed==1 and len(events)==0
    fresh=make_fx(double)(x)
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'post_grad_custom_post_pass':install}):
        compiled=compile_fx_inner(fresh,[x.clone()],is_inference=True)
finally:GraphLowering.save_output_code=staticmethod(orig_save)
after_compile=len(events);ai=[x.clone()];actual=compiled(list(ai));torch.testing.assert_close(actual,double(x));torch.testing.assert_close(ai[0],x)
assert actual[0].untyped_storage().data_ptr()!=ai[0].untyped_storage().data_ptr()
report['lowering_entry']={'count':count,'calls_after_apply':after_apply,'calls_after_compile':after_compile,'calls_after_runtime':len(events),'events':events,'output':actual[0].tolist(),'checks':'numeric, unchanged input, no input alias pass'}
assert (after_apply,after_compile,len(events))==(0,1,1) and events[0]['arg_type']=='TensorBox'
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
