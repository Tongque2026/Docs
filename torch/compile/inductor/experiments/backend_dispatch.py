"""真实CPU后端实例消费；设备创建/基类/组合委托为明确元数据与delegate fixture。"""
from pathlib import Path
import os,json,tempfile,types
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];O=R/'experiments/backend-dispatch';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=tempfile.mkdtemp(prefix='cache-',dir=O)
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1';os.environ['TORCH_WARM_POOL']='0'
import torch
from torch._inductor import config,ir
from torch._inductor.codegen import common as c
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
import torch._inductor.codegen.cuda_combined_scheduling as combined
import torch._inductor.scheduler as sm
import torch._inductor.utils as u
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from torch._inductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx
report={}
# 一次真实CPU编译，在原调度方法外采集对象复用，不替换生成算法。
real_events=[];instances=[];orig_get=sm.Scheduler.get_backend;orig_create=sm.Scheduler.create_backend
class LessonScheduling(CppScheduling):
    def __init__(self,scheduler):
        real_events.append({'kind':'construct','probe':scheduler is None});super().__init__(scheduler)
def get_backend(self,device):
    hit=device in self.backends;out=orig_get(self,device)
    if not any(out is item for item in instances):instances.append(out)
    real_events.append({'kind':'get','device':str(device),'hit':hit,'instance':next(i for i,b in enumerate(instances) if b is out)});return out
def create_backend(self,device):
    real_events.append({'kind':'create','device':str(device)});return orig_create(self,device)
c.init_backend_registration();saved=c.device_codegens['cpu'];c.register_backend_for_device('cpu',LessonScheduling,WrapperCodeGen,CppWrapperCpu)
x=torch.arange(6,dtype=torch.float32).reshape(2,3)-2;before=x.clone();fn=lambda x:(torch.relu(x+1)*2,)
try:
    with patch.object(sm.Scheduler,'get_backend',get_backend),patch.object(sm.Scheduler,'create_backend',create_backend),patch.object(GraphLowering,'save_output_code',side_effect=lambda code:(O/'cpu_output.py').write_text(code)),config.patch({'force_disable_caches':True,'cpp.simdlen':1,'triton.cudagraphs':False}):
        gm=make_fx(fn)(x);compiled=compile_fx_inner(gm,[x],is_inference=True)
    actual=compiled([x]);torch.testing.assert_close(actual,fn(x));assert torch.equal(x,before)
    assert len(instances)==1 and sum(e['kind']=='create' for e in real_events)==1
    assert any(e.get('hit') for e in real_events)
    assert actual[0].untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
    report['cpu']={'events':real_events,'backend_instances':len(instances),'output':actual[0].tolist(),'input_preserved':True,'output_aliases_input':False}
finally:c.device_codegens['cpu']=saved
# decode_device实际helper，GPU/PrivateUse1的当前编号提供者为fixture，无设备分配。
lookups=[]
def iface(device):
    lookups.append(device);return types.SimpleNamespace(Worker=types.SimpleNamespace(current_device=lambda:2))
with patch.object(u,'get_interface_for_device',side_effect=iface):
    decoded={name:str(u.decode_device(name)) for name in ['cpu','meta','cuda','xpu','privateuseone','cuda:1']}
    default=str(u.decode_device(None));specified=torch.device('cuda:1');assert u.decode_device(specified) is specified
assert lookups==['cuda','xpu','privateuseone']
report['decode']={'values':decoded,'default':default,'worker_lookups':lookups,'specified_same_object':True,'worker_fixture':True}
# 实际Scheduler方法，receiver与图状态是最小fixture；不生成GPU kernel。
class Receiver:
    create_backend=sm.Scheduler.create_backend
    get_backend=sm.Scheduler.get_backend
    def __init__(self):self.backends={}
class Backend:
    constructed=[]
    def __init__(self,scheduler):self.scheduler=scheduler;self.constructed.append(self)
class Replacement(Backend):pass
saved=dict(c.device_codegens)
for name in ['cpu','cuda','xpu','privateuseone']:c.register_backend_for_device(name,Backend,WrapperCodeGen)
added=[];graph=types.SimpleNamespace(add_device_info=lambda device:added.append(str(device)))
creation={}
with V.set_graph_handler(graph):
    receiver=Receiver()
    with patch.object(sm,'has_triton',return_value=True):
        a=receiver.get_backend(torch.device('cuda:0'));again=receiver.get_backend(torch.device('cuda:0'));b=receiver.get_backend(torch.device('cuda:1'))
        assert a is again and a is not b and a.scheduler is receiver
        c.register_backend_for_device('cuda',Replacement,WrapperCodeGen)
        stale=receiver.get_backend(torch.device('cuda:0'));fresh=Receiver().get_backend(torch.device('cuda:0'))
        assert stale is a and isinstance(fresh,Replacement)
    creation['cache']={'same_device_same_object':True,'indices_distinct':True,'registration_change_keeps_existing_instance':True,'new_receiver_uses_new_registration':True}
    # 显式缺失、GPU未归一、Triton缺失与能力分支，设备属性均fixture。
    for label,name,triton,major in [('missing','meta',True,8),('unnormalized','cuda',True,8),('old_cuda','cuda:0',False,6),('no_triton_cuda','cuda:0',False,8),('no_triton_xpu','xpu:0',False,8),('cpu_without_triton','cpu',False,8),('private_without_triton','privateuseone',False,8)]:
        r=Receiver();added.clear();props=types.SimpleNamespace(major=major,minor=1,name='fixture GPU')
        with patch.object(sm,'has_triton',return_value=triton),patch.object(torch.cuda,'get_device_properties',return_value=props) as properties:
            try:out=r.get_backend(torch.device(name))
            except Exception as e:
                creation[label]={'error':type(e).__name__,'message':str(e),'cache_size':len(r.backends),'added':list(added),'property_calls':properties.call_count}
            else:creation[label]={'kind':type(out).__name__,'cache_size':len(r.backends),'added':list(added),'property_calls':properties.call_count}
    assert creation['unnormalized']['error']=='AssertionError' and not creation['unnormalized']['added']
    assert creation['missing']['error']=='RuntimeError' and creation['missing']['added']==['meta']
    assert 'too old' in creation['old_cuda']['message']
    assert 'working triton' in creation['no_triton_cuda']['message'] and 'working triton' in creation['no_triton_xpu']['message']
    assert creation['no_triton_xpu']['property_calls']==0
    assert creation['cpu_without_triton']['kind']==creation['private_without_triton']['kind']=='Backend'
    # 构造失败没有缓存结果，但设备记录已发生；重试再次创建。
    attempts=[]
    def failing(scheduler):attempts.append(1);raise ValueError('fixture constructor failure')
    c.register_backend_for_device('cpu',failing,WrapperCodeGen);r=Receiver();added.clear()
    for _ in range(2):
        try:r.get_backend(torch.device('cpu'))
        except ValueError:pass
    assert len(attempts)==2 and not r.backends and added==['cpu','cpu']
    creation['constructor_failure']={'attempts':2,'cache_size':0,'device_record_calls':list(added)}
c.device_codegens.clear();c.device_codegens.update(saved);report['creation_protocol']=creation
# Base默认契约：融合只验证转交，实际merge由fixture替代。
base=sm.BaseScheduling();errors=[]
for name,args in [('can_fuse_vertical',(None,None)),('can_fuse_horizontal',(None,None)),('group_fn',([],)),('codegen_template',(None,[])),('codegen_node',(None,)),('codegen_sync',()),('flush',()),('benchmark_fused_nodes',([],)),('benchmark_combo_kernel',([],))]:
    try:getattr(base,name)(*args)
    except NotImplementedError:errors.append(name)
assert len(errors)==9 and base.get_backend_features(torch.device('cpu'))==() and base.ready_to_flush() is False and base.get_fusion_pair_priority(None,None)==0
try:sm.BaseScheduling(None)
except TypeError:ctor='TypeError'
else:raise AssertionError('base does not accept Scheduler')
node=lambda foreach:types.SimpleNamespace(is_foreach=lambda:foreach)
with patch.object(sm.FusedSchedulerNode,'fuse',return_value='ordinary-merge') as ordinary,patch.object(sm.ForeachKernelSchedulerNode,'fuse',return_value='foreach-merge') as foreach:
    branches=[base.fuse(node(False),node(False)),base.fuse(node(True),node(False)),base.fuse(node(False),node(True))]
assert branches==['ordinary-merge','foreach-merge','foreach-merge']
report['base']={'unimplemented':errors,'features':[],'ready_to_flush':False,'fusion_priority':0,'scheduler_constructor_error':ctor,'merge_branches':branches,'merge_fixture':True}
# 真实组合对象与方法，三个底层delegate为记录fixture；返回标签绝非benchmark时间。
log=[]
class Delegate:
    def __init__(self,tag,owner):self.tag=tag;self.owner=owner;self.vertical=False
    def is_cuda_cpp_template(self,node):return getattr(node,'cuda',False)
    def is_rocm_cpp_template(self,node):return getattr(node,'rocm',False)
    def can_fuse_vertical(self,*args):log.append(self.tag+'.vertical');return self.vertical
    def can_fuse_horizontal(self,*args):log.append(self.tag+'.horizontal');return False
    def __getattr__(self,name):
        def call(*args,**kw):log.append(self.tag+'.'+name);return self.tag+'.'+name
        return call
owner=object()
with patch.object(combined,'TritonScheduling',side_effect=lambda s:Delegate('triton',s)),patch.object(combined,'CUDACPPScheduling',side_effect=lambda s:Delegate('cuda',s)),patch.object(combined,'ROCmCPPScheduling',side_effect=lambda s:Delegate('rocm',s)):
    backend=combined.CUDACombinedScheduling(owner)
assert all(d.owner is owner for d in [backend._triton_scheduling,backend._cuda_cpp_scheduling,backend._rocm_cpp_scheduling])
nodes={'ordinary':types.SimpleNamespace(cuda=False,rocm=False),'cuda':types.SimpleNamespace(cuda=True,rocm=False),'rocm':types.SimpleNamespace(cuda=False,rocm=True),'both':types.SimpleNamespace(cuda=True,rocm=True)}
choices={name:backend.choose_node_backend(n).tag for name,n in nodes.items()}
assert choices=={'ordinary':'triton','cuda':'cuda','rocm':'rocm','both':'cuda'}
templates={name:backend.codegen_template(n,[]) for name,n in nodes.items()}
assert templates=={name:tag+'.codegen_template' for name,tag in choices.items()}
rejects=[]
for name in ['cuda','rocm']:
    try:backend.codegen_template(nodes[name],[object()])
    except AssertionError:rejects.append(name)
assert rejects==['cuda','rocm']
assert backend.codegen_template(nodes['ordinary'],[object()])=='triton.codegen_template'
forwarded={}
for name,args in [('get_backend_features',(torch.device('cuda:0'),)),('group_fn',(([2,3],[]),)),('codegen_node',(nodes['cuda'],)),('codegen_sync',()),('flush',()),('codegen_combo_kernel',('nodes',)),('benchmark_fused_nodes',('nodes',)),('generate_kernel_code_from_nodes',('nodes',True)),('benchmark_combo_kernel',('nodes',))]:
    forwarded[name]=getattr(backend,name)(*args);assert forwarded[name]=='triton.'+name
log.clear();backend._cuda_cpp_scheduling.vertical=True;assert backend.can_fuse_vertical(nodes['ordinary'],nodes['cuda']) is True;short=list(log)
log.clear();backend._cuda_cpp_scheduling.vertical=False;backend._triton_scheduling.vertical=True;assert backend.can_fuse_vertical(nodes['ordinary'],nodes['rocm']) is True;fallback=list(log)
assert short==['cuda.vertical'] and fallback==['cuda.vertical','triton.vertical']
log.clear();backend.can_fuse_horizontal(nodes['ordinary'],nodes['cuda']);hcuda=list(log)
log.clear();backend.can_fuse_horizontal(nodes['ordinary'],nodes['rocm']);hrocm=list(log)
assert hcuda==['cuda.horizontal'] and hrocm==['triton.horizontal']
# 直接真实CUDA C++水平谓词仍继承NotImplementedError；不伪称注释中的总False。
real_cuda=combined.CUDACPPScheduling(None);backend._cuda_cpp_scheduling=real_cuda
n=sm.SchedulerNode.__new__(sm.SchedulerNode);n.node=ir.CUDATemplateBuffer.__new__(ir.CUDATemplateBuffer)
try:backend.can_fuse_horizontal(n,n)
except NotImplementedError:real_horizontal='NotImplementedError'
else:raise AssertionError('real CUDA C++ horizontal missing implementation')
report['combined_protocol']={'choices':choices,'templates':templates,'nonempty_epilogue_rejects':rejects,'forwarded':forwarded,'vertical_short':short,'vertical_fallback':fallback,'horizontal_cuda':hcuda,'horizontal_rocm':hrocm,'real_cuda_horizontal':real_horizontal,'ready_to_flush_inherited':backend.ready_to_flush(),'priority_inherited':backend.get_fusion_pair_priority(None,None),'delegate_fixtures':True,'benchmark_values':'dispatch labels only; no timing or GPU execution'}
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
