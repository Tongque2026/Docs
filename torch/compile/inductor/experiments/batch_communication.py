"""真实 CPU 批处理改写与不执行 collective 的图结构实验。"""
from pathlib import Path
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.fx_passes import group_batch_fusion as gb,ddp_fusion as ddp,micro_pipeline_tp as tp
from torch._inductor import config
from torch._inductor.virtualized import V
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/batch-communication';OUT.mkdir(exist_ok=True)
aten=torch.ops.aten;c10d=torch.ops._c10d_functional
clone=lambda ins:[t.clone() for t in ins]
def fn(*xs):return tuple(x.relu() for x in xs)
def storage_matrix(ts):return [[a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr() for b in ts] for a in ts]
report={}
for name,count,options,mixed in [('default_five',5,{},False),('below_threshold',4,{},False),('mixed_shapes',5,{},True),('max_two',5,{'min_fuse_set_size':2,'max_fuse_set_size':2},False),('require_fbgemm',5,{'require_fbgemm':True},False)]:
    inputs=[torch.arange(6,dtype=torch.float32).reshape(2,3)-i for i in range(count)]
    if mixed:inputs[-1]=torch.tensor([-1.,0.,1.])
    gm=make_fx(fn)(*clone(inputs));mode=fake_tensor_prop(gm,clone(inputs));before=str(gm.graph)
    with config.patch({'post_grad_fusion_options':{'batch_aten_relu':options}}),V.set_fake_mode(mode):gb.group_batch_fusion_passes(gm.graph,pre_grad=False)
    gm.graph.lint();gm.recompile();ai,ei=clone(inputs),clone(inputs);actual=gm(*ai);expected=fn(*ei)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(ai,ei):torch.testing.assert_close(a,e)
    assert not any(o.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for o in actual for i in ai)
    targets=[str(n.target) for n in gm.graph.nodes if n.op=='call_function']
    report[name]={'has_fbgemm':gb.has_fbgemm,'targets':targets,'relu_count':targets.count('aten.relu.default'),'cat_count':targets.count('aten.cat'),'outputs':[o.tolist() for o in actual],'output_shapes':[list(o.shape) for o in actual],'offsets':[o.storage_offset() for o in actual],'output_storage_before':storage_matrix(expected),'output_storage_after':storage_matrix(actual),'checks':'numeric, input unchanged, output not aliasing inputs pass'}
    (OUT/f'{name}_before.txt').write_text(before+'\n');(OUT/f'{name}_after.txt').write_text(str(gm.graph)+'\n')
assert report['default_five']['relu_count']==1 and report['default_five']['cat_count']==1
assert report['below_threshold']['relu_count']==4 and report['mixed_shapes']['relu_count']==5
assert report['max_two']['relu_count']==3 and report['max_two']['cat_count']==2
assert report['require_fbgemm']['relu_count']==(1 if gb.has_fbgemm else 5)
assert report['default_five']['offsets']==[0,6,12,18,24]
# 直接运行真实独立子集算法，不把前后依赖节点放入同一个 batch。
def dep_fn(x,y):
    a=x.relu();b=a.relu();c=y.relu();return (b,c)
gm=make_fx(dep_fn)(torch.ones(2),torch.ones(2));nodes=[n for n in gm.graph.nodes if n.target==aten.relu.default]
opts={**gb.default_graph_search_options,'min_fuse_set_size':2,'max_fuse_set_size':3}
subsets=[list(g) for g in gb.find_independent_subset_greedy(nodes,opts)]
report['independent_subsets']={'input':[n.name for n in nodes],'subsets':[[n.name for n in g] for g in subsets]}
assert [[n.name for n in g] for g in subsets]==[['relu','relu_2']]
# 真实通信图容器和真实 schedule_comm_wait；不执行任何通信 op。
g=torch.fx.Graph();a=g.placeholder('a');y=g.placeholder('y');a.meta['tensor_meta']=_extract_tensor_metadata(torch.zeros(2,3))
comm=g.call_function(c10d.all_reduce.default,(a,'sum','teaching_group'))
wait=g.call_function(c10d.wait_tensor.default,(comm,));other=g.call_function(aten.relu.default,(y,));out=g.call_function(aten.add.Tensor,(wait,other));g.output((out,));gm=torch.fx.GraphModule({},g)
before=[n.name for n in g.nodes];blocks=ddp.get_all_comm_blocks(g,(c10d.all_reduce.default,))
ddp.schedule_comm_wait(g);g.lint();after=[n.name for n in g.nodes]
report['wait_schedule']={'before':before,'after':after,'blocks':[{'comm':b.comm_node.name,'waits':[n.name for n in b.wait_nodes],'outputs':[n.name for n in b.outputs]} for b in blocks],'execution':'graph structure only; no collective executed'}
assert before.index(wait.name)<before.index(other.name) and after.index(other.name)<after.index(wait.name)<after.index(out.name)
# 顶层 dispatcher 按配置顺序调用，并按签名传递 bucket_size_mb。
events=[]
def first(graph,bucket_size_mb):events.append({'pass':'first','bucket_size_mb':bucket_size_mb})
def second(graph):events.append({'pass':'second'})
ddp.fuse_ddp_communication(g,[first,'schedule_comm_wait',second],bucket_size_mb=7)
report['ddp_dispatch']={'events':events,'execution':'dispatcher and graph scheduling only'}
assert events==[{'pass':'first','bucket_size_mb':7},{'pass':'second'}]
# 直接构造合法 FX 关系，运行真实 TP finder，不创建 process group。
g=torch.fx.Graph();shard=g.placeholder('shard');weight=g.placeholder('weight');ag=g.call_function(c10d.all_gather_into_tensor.default,(shard,2,'teaching_group'));wait=g.call_function(c10d.wait_tensor.default,(ag,));mm=g.call_function(aten.mm.default,(wait,weight));g.output((mm,));gm=torch.fx.GraphModule({},g)
found=tp.find_all_gather_patterns(g)
report['tp_find']={'count':len(found),'matches':[{'shard':m.shard_node.name,'collective':m.ag_node.name,'result':m.res_node.name,'gather_dim':m.gather_dim,'group':m.group_name} for m in found],'execution':'pattern matching only; no collective or fused kernel executed'}
assert len(found)==1 and found[0].gather_dim==0
(OUT/'tp_pattern.txt').write_text(str(g)+'\n')
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
