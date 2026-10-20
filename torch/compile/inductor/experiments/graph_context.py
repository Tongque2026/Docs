"""真实 GraphLowering 的上下文、设备与符号状态；不生成设备 kernel。"""
from pathlib import Path
import json,threading
import torch,sympy
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V,Virtualized,NullHandler
from torch._inductor.codegen.common import BackendFeature
from torch._inductor import config
R=Path(__file__).resolve().parents[1];O=R/'experiments/graph-context';O.mkdir(exist_ok=True)
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
gm=make_fx(lambda x:(x+1,))(x)
def new(**kw):return GraphLowering(gm,[x],layout_opt=False,**kw)
results={}
a=new(name='outer',cpp_wrapper=True,aot_mode=True,is_inference=True)
b=new()
assert a._shape_env is not b._shape_env and a.sizevars.shape_env is a._shape_env
assert not a.reuse_shape_env and a._shape_env.runtime_asserts_frozen
assert a.wrapper_code is None and a.scheduler is None
assert not a.device_types and not a.device_idxs
results['fresh']={'independent_shape_env':True,'runtime_asserts_frozen':True,'initial_device_types':list(a.device_types),'wrapper':None,'scheduler':None,'reuse_shape_env':a.reuse_shape_env}
# Deferred assertions are built with the real ShapeEnv API before freezing.
se=ShapeEnv();u=se.create_unbacked_symint();v=se.create_unbacked_symint()
se.defer_runtime_assert(sympy.Eq(u.node.expr+v.node.expr,7),'sum seven')
c=new(shape_env=se)
key=next(iter(se.deferred_runtime_asserts))
assert c._shape_env is se and c.reuse_shape_env and c.ras_by_symbol is not se.deferred_runtime_asserts
assert c.ras_by_symbol[key] is se.deferred_runtime_asserts[key]
count=len(se.deferred_runtime_asserts);c.ras_by_symbol.pop(key)
assert len(se.deferred_runtime_asserts)==count
try:se.defer_runtime_assert(sympy.Eq(u.node.expr*v.node.expr,3),'product three')
except AssertionError as e:error=type(e).__name__
else:raise AssertionError('new deferred assert accepted')
assert se.defer_runtime_assert(sympy.true,'known true') is sympy.true
results['assert_snapshot']={'same_shape_env':True,'copied_dict':True,'shared_value_lists':True,'source_keys_survive_pop':True,'new_unknown_assert_error':error,'known_true_allowed':True,'key':str(key)}
# Symbol expression conversion: reused ShapeEnv takes the existing fake symbols.
se2=ShapeEnv();mode=FakeTensorMode(shape_env=se2)
f=mode.from_tensor(x,static_shapes=False)
before_symbols=dict(se2.var_to_val)
d=new(shape_env=se2)
sizes,strides=d.symbolic_sizes_strides(f)
assert sizes==[z.node.expr if isinstance(z,torch.SymInt) else sympy.sympify(z) for z in f.size()]
assert se2.var_to_val==before_symbols
results['reuse_symbols']={'sizes':list(map(str,sizes)),'strides':list(map(str,strides)),'source_sizes':list(map(str,f.size())),'symbols':len(se2.var_to_val),'same_expressions':True}
# Fresh ShapeEnv dynamic helper duck-shapes equal concrete dimensions across calls.
e=new();ss1,st1=e.symbolic_sizes_strides(torch.empty(2,3));ss2,st2=e.symbolic_sizes_strides(torch.empty(2,5))
assert ss1[0]==ss2[0] and ss1[1]!=ss2[1]
static=e.static_sizes_strides(torch.empty(3,2).t())
assert static==([sympy.Integer(2),sympy.Integer(3)],[sympy.Integer(1),sympy.Integer(2)])
results['fresh_symbols']={'first_sizes':list(map(str,ss1)),'second_sizes':list(map(str,ss2)),'first_strides':list(map(str,st1)),'same_size_shared_symbol':True,'static_size_stride':[[int(z) for z in q] for q in static]}
# Real device-info method, with metadata-only torch.device objects.
nodes=list(gm.graph.nodes);n0,n1=nodes[:2]
with V.set_graph_handler(a):
 a.add_device_info(torch.device('cpu'))
 with a.set_current_node(n0):
  a.add_device_info(torch.device('cuda',0));a.add_device_info(torch.device('cuda',1))
 with a.set_current_node(n1):a.add_device_info(torch.device('cuda',0));a.add_device_info(torch.device('xpu',0))
 assert a.current_node is None
assert a.device_node_mapping[torch.device('cuda',0)] is n0
results['device_registration']={'types':list(a.device_types),'indices':list(a.device_idxs),'first_node_by_device':{str(k):v.name for k,v in a.device_node_mapping.items()},'cpu_without_node_recorded':torch.device('cpu') not in a.device_node_mapping,'scope':'device metadata only; no CUDA/XPU allocation or runtime operation'}
mix=new()
with V.set_graph_handler(mix):
 for dev in ['cpu','cuda:0','xpu:0']:mix.add_device_info(torch.device(dev))
 try:mix.init_wrapper_code()
 except AssertionError as exc:mix_error=str(exc)
 else:raise AssertionError('mixed accelerator types accepted')
assert 'Does not support mixing' in mix_error
results['mixed_devices']={'types':list(mix.device_types),'error':'AssertionError','message':mix_error,'scope':'real wrapper selection rejection; metadata devices only'}
# Constructor sharing for const_module differs from separately lowered subgraphs.
a.constants['k']=torch.tensor(1.);a.allocated_constant_name['k']='constant_k'
main=new(const_module=a,const_output_index={'folded':0})
assert main.device_types is a.device_types and main.device_idxs is a.device_idxs
assert main.constants is a.constants and main.allocated_constant_name is a.allocated_constant_name
assert main.operations is not a.operations and list(main.folded_constants)==['folded']
results['constant_sharing']={'same_devices':True,'same_constants':True,'same_constant_names':True,'separate_operations':True,'folded':list(main.folded_constants)}
sub=a.make_subgraph(gm,[x],'branch')
assert sub._shape_env is a._shape_env and sub.reuse_shape_env and sub.name=='outer_branch'
assert sub.constants is not a.constants and sub.device_types is not a.device_types
assert sub.cpp_wrapper and sub.aot_mode and sub.is_inference
results['subgraph']={'name':sub.name,'same_shape_env':True,'separate_constants':True,'separate_devices':True,'modes':[sub.cpp_wrapper,sub.aot_mode,sub.is_inference],'qualified_name':sub.qualify_name('buf0'),'unnamed_qualify':b.qualify_name('buf0')}
# Nested contexts, exceptions, and graph-local current_node vs virtualized current_node.
prior=V.graph;prior_node=V.current_node
def restored(value,old):return isinstance(value,NullHandler) if isinstance(old,NullHandler) else value is old
with V.set_graph_handler(a):
 assert V.graph is a
 try:
  with V.set_graph_handler(b),a.set_current_node(n0),V.set_current_node(n1):
   assert V.graph is b and a.current_node is n0 and V.current_node is n1
   raise ValueError('restore')
 except ValueError:pass
 assert V.graph is a and a.current_node is None and restored(V.current_node,prior_node)
assert restored(V.graph,prior)
results['scope_restore']={'nested_graph_restored':True,'exception_restored':True,'distinct_current_nodes':True,'outer_restored':True}
# _set_handler changes the handler immediately, before entering returned context.
vh=Virtualized('lesson_context_probe',NullHandler);obj=object();ctx=vh._set_handler(obj)
assert vh._get_handler() is obj
with ctx:assert vh._get_handler() is obj
assert isinstance(vh._get_handler(),NullHandler)
results['setter_timing']={'installed_before_enter':True,'restored_after_exit':True,'missing_default':'NullHandler'}
barrier=threading.Barrier(3);seen={}
def worker(tag,g):
 with V.set_graph_handler(g):
  barrier.wait();seen[tag]=V.graph is g;barrier.wait()
threads=[threading.Thread(target=worker,args=(tag,g)) for tag,g in [('a',a),('b',b)]]
for th in threads:th.start()
with V.set_graph_handler(c):
 barrier.wait();main_is_c=V.graph is c;barrier.wait()
for th in threads:th.join()
assert all(seen.values()) and main_is_c
results['thread_local']={'workers':seen,'main_kept_own_graph':main_is_c}
# Feature cache per GraphLowering; real method asks an injected counting provider.
# No new backend registered; provider fixture isolates per-graph memoization.
import torch._inductor.graph as graph_module
orig=graph_module.get_backend_features;calls=[]
def features(dev):calls.append(dev);return {BackendFeature.FOREACH}
graph_module.get_backend_features=features
try:
 cached=new();assert cached.has_feature(torch.device('cpu'),BackendFeature.FOREACH)
 assert cached.has_feature(torch.device('cpu'),BackendFeature.FOREACH)
 other=new();assert other.has_feature(torch.device('cpu'),BackendFeature.FOREACH)
 try:cached.has_feature(torch.device('cpu'),'FOREACH')
 except AssertionError:bad='AssertionError'
 else:raise AssertionError('invalid feature accepted')
 assert calls==['cpu','cpu']
finally:graph_module.get_backend_features=orig
results['feature_cache']={'provider_calls':calls,'invalid_feature':bad,'scope':'real has_feature and constructor cache with explicit provider fixture'}
# Run actual graph lowering, showing used vs unused placeholder device registration.
mode2=FakeTensorMode();real_gm=make_fx(lambda x,y:(x+1,))(x,torch.ones(4))
for n in real_gm.graph.nodes:
 if n.op=='placeholder':n.meta['val']=mode2.from_tensor(x if n.target=='x_1' else torch.ones(4))
# use existing fake propagation helper to populate all node metadata
from torch._inductor.compile_fx import fake_tensor_prop
mode2=fake_tensor_prop(real_gm,[x,torch.ones(4)])
with V.set_fake_mode(mode2):
 run_graph=GraphLowering(real_gm,[x,torch.ones(4)],layout_opt=False)
 with V.set_graph_handler(run_graph):run_graph.run(x,torch.ones(4))
assert len(run_graph.graph_inputs)==2 and list(run_graph.device_types)==['cpu']
assert run_graph.device_node_mapping[torch.device('cpu')].name=='x_1'
results['lowering']={'inputs':list(run_graph.graph_inputs),'devices':list(run_graph.device_types),'first_device_node':run_graph.device_node_mapping[torch.device('cpu')].name,'operations':len(run_graph.operations),'scope':'real CPU IR lowering; no kernel compilation/execution'}
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2))
