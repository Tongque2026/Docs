"""真实稳定拓扑排序的状态、入口更新与边界；不修改产品源码。"""
from pathlib import Path
import json,sys,operator
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.pattern_matcher import stable_topological_sort,_args
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/order-entry';OUT.mkdir(exist_ok=True)
a=torch.ops.aten;x=torch.arange(4,dtype=torch.float32);y=x+10

def snapshot(frame):
    v=frame.f_locals
    return {'node':getattr(v.get('node'),'name',None),'pending':[n.name for n in v.get('pending',[])],'ready':sorted(n.name for n in v.get('ready',set())),'waiting':{k.name:[n.name for n in ns] for k,ns in v.get('waiting',{}).items()},'waiting_for':[n.name for n in v.get('waiting_for',[])],'cursor':getattr(v.get('cursor'),'name',None)}
def traced_sort(graph):
    events=[];old=sys.gettrace()
    def trace(frame,event,arg):
        if frame.f_code is stable_topological_sort.__code__ and event=='line' and frame.f_lineno in [1918,1922,1924,1932]:
            events.append({'line':frame.f_lineno,**snapshot(frame)})
        return trace
    try:sys.settrace(trace);stable_topological_sort(graph)
    finally:sys.settrace(old)
    return events
names=lambda g:[n.name for n in g.nodes]
report={}
for name,reverse_producers in [('late_last_arg',False),('last_arg_ready_first',True)]:
    gm=make_fx(lambda x,y:(-x+y.relu(),))(x,y)
    neg=next(n for n in gm.graph.nodes if n.target==a.neg.default);relu=next(n for n in gm.graph.nodes if n.target==a.relu.default);add=next(n for n in gm.graph.nodes if n.target==a.add.Tensor)
    neg.prepend(add)
    if reverse_producers:neg.prepend(relu)
    before=names(gm.graph);original_nodes=set(gm.graph.nodes);tag=object();neg.meta['teaching_tag']=tag
    events=traced_sort(gm.graph);gm.graph.lint();gm.recompile();torch.testing.assert_close(gm(x,y),(-x+y.relu(),))
    assert set(gm.graph.nodes)==original_nodes and neg.meta['teaching_tag'] is tag
    after=names(gm.graph);again=traced_sort(gm.graph);assert names(gm.graph)==after
    report[name]={'before':before,'after':after,'events':events,'second_events':again,'same_node_objects_and_metadata':True,'output':gm(x,y)[0].tolist()}
# 名字中的 stable 不保证已无效顺序中所有无依赖对的相对次序。
gm=make_fx(lambda x:((x+1)*2,x.sin()))(x);add=next(n for n in gm.graph.nodes if n.target==a.add.Tensor);mul=next(n for n in gm.graph.nodes if n.target==a.mul.Tensor);sin=next(n for n in gm.graph.nodes if n.target==a.sin.default)
add.prepend(mul);add.prepend(sin);before=names(gm.graph);traced_sort(gm.graph);after=names(gm.graph);gm.graph.lint();gm.recompile();torch.testing.assert_close(gm(x),((x+1)*2,x.sin()))
report['incomparable_pair']={'before':before,'after':after,'pair_before':before.index(mul.name)<before.index(sin.name),'pair_after':after.index(mul.name)<after.index(sin.name),'outputs':[o.tolist() for o in gm(x)]}
assert report['incomparable_pair']['pair_before'] and not report['incomparable_pair']['pair_after']
# map_arg 收集 kwargs 和嵌套结构中的 Node，保留重复引用。
g=torch.fx.Graph();p=g.placeholder('p');q=g.placeholder('q');z=g.placeholder('z');n=g.call_function(operator.add,(p,[q,p]),{'nested':{'value':z},'constant':7});g.output((p,))
report['nested_args']={'dependencies':[v.name for v in _args(n)]};assert report['nested_args']['dependencies']==['p','q','p','z']
# 图和已生成 forward 独立；lint 对同一图的数值意图不作判断。
gm=make_fx(lambda x:(x+1,))(x);add=next(n for n in gm.graph.nodes if n.target==a.add.Tensor);old_code=gm.code;add.args=(add.args[0],3)
gm.graph.lint();old=gm(x)[0];interpreted=torch.fx.Interpreter(gm).run(x)[0];gm.recompile();new=gm(x)[0];gm.graph.lint()
assert torch.equal(old,x+1) and torch.equal(interpreted,x+3) and torch.equal(new,x+3)
report['forward_refresh']={'before_recompile':old.tolist(),'interpreter_current_graph':interpreted.tolist(),'after_recompile':new.tolist(),'code_changed':gm.code!=old_code,'lint_before_and_after':'pass'}
# 不存在合法数据顺序的环和外图依赖，都留下 waiting 并由末尾断言拒绝。
for case in ['cycle','foreign_dependency']:
    gm=make_fx(lambda x:((x+1)*2,))(x);add=next(n for n in gm.graph.nodes if n.target==a.add.Tensor);mul=next(n for n in gm.graph.nodes if n.target==a.mul.Tensor)
    if case=='cycle':add.args=(mul,1)
    else:other=torch.fx.Graph();foreign=other.placeholder('foreign');add.args=(foreign,1)
    try:traced_sort(gm.graph)
    except AssertionError as e:report[case]={'error':type(e).__name__}
    else:raise AssertionError(case)
# lint 对外图引用给出自己的结构错误。
gm=make_fx(lambda x:(x+1,))(x);add=next(n for n in gm.graph.nodes if n.target==a.add.Tensor);other=torch.fx.Graph();foreign=other.placeholder('foreign');add.args=(foreign,1)
try:gm.graph.lint()
except RuntimeError as e:report['lint_ownership']={'error':type(e).__name__,'message_prefix':str(e).splitlines()[0]}
else:raise AssertionError('lint must reject foreign ownership')
# 数据拓扑不包含任意副作用顺序；构造两个都合法的链表顺序作对照。
g=torch.fx.Graph();px=g.placeholder('x');py=g.placeholder('y');tmp=g.call_function(a.add.Tensor,(px,0));copy=g.call_function(a.copy_.default,(tmp,py));read=g.call_function(a.add.Tensor,(tmp,1));g.output((read,));gm=torch.fx.GraphModule({},g)
original=gm(x,y)[0].clone();copy.prepend(read);before=names(g);stable_topological_sort(g);after=names(g);g.lint();gm.recompile();changed=gm(x,y)[0]
assert before==after and torch.equal(original,y+1) and torch.equal(changed,x+1)
report['mutation_boundary']={'changed_order_before_sort':before,'after_sort':after,'original_result':original.tolist(),'changed_result':changed.tolist(),'scope':'intentional invalid semantic reordering; both node orders satisfy explicit data dependencies'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
