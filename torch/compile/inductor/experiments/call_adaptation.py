"""真实CPU编译/调用，真实CPU对齐helper，以及明确GPU描述/捕获factory fixture。"""
import os,json,tempfile,contextlib,types
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];O=R/'experiments/call-adaptation';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=tempfile.mkdtemp(prefix='cache-',dir=O)
os.environ['TORCH_WARM_POOL']='0';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch._inductor import config
import torch._inductor.compile_fx as cf
import torch._inductor.codecache as cc
import torch._inductor.utils as u
import torch._inductor.cudagraph_trees as trees
from torch._inductor.cudagraph_utils import CudagraphCachedInfo,BoxedDeviceIndex,check_for_mutation_ignore_cuda_graph_managed_tensor
from torch._inductor.graph import GraphLowering
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
from torch._guards import TracingContext,tracing
report={'cpu':{},'alignment':{},'policy':{},'deferred_protocol':{}}
base=torch.arange(7,dtype=torch.float32);x=base[1:].reshape(2,3)
assert x.data_ptr()%16==4
for label,fn in [('pure',lambda x:(x*2+1,)),('mutation',lambda x:(x.add_(3),))]:
    events=[];orig_post=cc.FxGraphCache.post_compile
    def post(graph,inputs,boxed):
        before=graph.current_callable
        out=orig_post(graph,inputs,boxed)
        events.append({'checks':list(graph.inputs_to_check),'boxed_after':bool(boxed),'disabled_reason':graph.disabled_cudagraphs_reason,'same_callable':graph.current_callable is before,'device_types':sorted(graph.device_types)})
        return out
    gm=make_fx(fn)(x.clone());flag=u.BoxedBool(True)
    with patch.object(cc.FxGraphCache,'post_compile',side_effect=post),patch.object(GraphLowering,'save_output_code',side_effect=lambda code:(O/(label+'_output.py')).write_text(code)),config.patch({'force_disable_caches':True,'cpp.simdlen':1,'triton.cudagraphs':True}):
        compiled=cf.compile_fx_inner(gm,[x],is_inference=True,cudagraphs=flag)
    value=torch.cat((torch.zeros(1),x.reshape(-1)))[1:].reshape(2,3);assert value.data_ptr()%16==4
    before=value.clone();args=[value];actual=compiled(args)
    if label=='pure':
        torch.testing.assert_close(actual[0],before*2+1);assert torch.equal(value,before)
    else:
        torch.testing.assert_close(value,before+3);assert actual[0].untyped_storage().data_ptr()==value.untyped_storage().data_ptr()
    assert not flag and events[0]['checks']==[]
    report['cpu'][label]={'events':events,'input_mod16':value.data_ptr()%16,'output':actual[0].tolist(),'input_after':value.tolist(),'input_preserved':torch.equal(value,before),'output_aliases_input':actual[0].untyped_storage().data_ptr()==value.untyped_storage().data_ptr(),'args_consumed':args==[]}
# 对齐helper仅使用真实CPU存储；显式指定检查idx，不冒充CPU默认会选择它。
original=torch.arange(13,dtype=torch.float32)[1:].reshape(3,4).t();old=original.clone();args=[original];seen=[]
def receive(args):
    seen.append({'shape':list(args[0].shape),'stride':list(args[0].stride()),'aligned':args[0].data_ptr()%u.ALIGNMENT==0,'same_tensor':args[0] is original});return args[0]+1
wrapped=u.align_inputs_from_check_idxs(receive,[0]);actual=wrapped(args)
torch.testing.assert_close(actual,old+1)
assert args[0] is not original and args[0].stride()==original.stride() and torch.equal(original,old)
assert u.align_inputs_from_check_idxs(receive,[]) is receive
aligned=torch.ones(4);aligned_args=[aligned];u.copy_misaligned_inputs(aligned_args,[0]);assert aligned_args[0] is aligned
static=[0,1,2];filtered=u.remove_unaligned_input_idxs([aligned,original,7],static);assert filtered==[0]
report['alignment']['real_cpu_helper']={'checks_for_actual_cpu':list(cf.get_input_idxs_to_check([original],[])),'before_mod16':original.data_ptr()%16,'seen':seen,'replaced_list_item':True,'original_preserved':True,'empty_checks_same_callable':True,'aligned_tensor_same_object':True,'filtered_static':filtered}
# helper没有通用mutation写回；这里只演示显式CPU helper调用，不作为生成器有效性证据。
mutation_original=torch.arange(5,dtype=torch.float32)[1:];before=mutation_original.clone();margs=[mutation_original]
def mutate_replacement(args):args[0].add_(5);return args[0]
changed=u.align_inputs_from_check_idxs(mutate_replacement,[0])(margs)
assert torch.equal(mutation_original,before) and torch.equal(changed,before+5)
report['alignment']['explicit_helper_mutation']={'original_preserved':True,'replacement_mutated':True,'no_writeback':True,'forced_helper_only':True}
# 布局/设备描述对照，无GPU分配。
mode=FakeTensorMode();meta=torch.empty(10,device='meta')
fc=FakeTensor(mode,meta.as_strided((2,3),(3,1),0),torch.device('cuda:0'))
fm=FakeTensor(mode,meta.as_strided((2,3),(3,1),1),torch.device('cuda:0'))
with config.patch({'assume_aligned_inputs':False}):
    nonstatic=cf.get_input_idxs_to_check([fc,fm,7],[]);static_check=cf.get_input_idxs_to_check([fc,fm],[0])
with config.patch({'assume_aligned_inputs':True}):assumed=cf.get_input_idxs_to_check([fc,fm],[])
assert list(nonstatic)==[0] and list(static_check)==[] and list(assumed)==[0,1]
report['alignment']['device_metadata']={'nonstatic':list(nonstatic),'static':list(static_check),'assume_aligned':list(assumed),'is_gpu':{d:u.is_gpu(d) for d in ['cpu','cuda','xpu','privateuseone']},'fake_only':True}
# 实际TracingContext接口回填简单输出stride；不假造符号环境。
ctx=TracingContext(FakeTensorMode());ctx.output_strides=[]
with tracing(ctx):u.set_tracing_context_output_strides([],types.SimpleNamespace(output_strides=[('3','1'),None]))
assert ctx.output_strides==[(3,1),None]
report['policy']['stride_context']={'output_strides':ctx.output_strides}
def model(args):
    out=args[0]+1;args.clear();return (out,)
def graph(reasons=(),disabled=None,backward=False):
    return types.SimpleNamespace(current_callable=model,cudagraph_info=CudagraphCachedInfo((),[],list(reasons)),inputs_to_check=[0],boxed_forward_device_index=BoxedDeviceIndex(2),fx_kwargs={'is_inference':not backward,'is_backward':backward,'static_input_idxs':[]},device_idxs={2},device_types={'cuda'},constants={},mutated_input_idxs=set(),disabled_cudagraphs_reason=disabled,output_strides=[('3','1')])
# post_compile接受真实代码路径，compiled描述与cudagraph工厂是明确fixture。
for label,reason,disabled in [('cached_disabled',[], 'fixture prior disable'),('failed_gate',['incompatible ops'],None),('enabled',[],None)]:
    g=graph(reason,disabled);flag=u.BoxedBool(True);factories=[]
    def factory(fn,**kwargs):
        factories.append({'device_index':kwargs['device_index'],'static_input_idxs':list(kwargs['static_input_idxs'])})
        def call(args):return fn(args)
        return call
    with patch.object(cf,'cudagraphify',side_effect=factory):result=cc.FxGraphCache.post_compile(g,[],flag)
    assert result is g
    args=[original];out=g.current_callable(args)[0];torch.testing.assert_close(out,original+1)
    report['policy'][label]={'boxed':bool(flag),'factory_calls':factories,'callable_changed':g.current_callable is not model,'output_correct':True}
assert not report['policy']['cached_disabled']['boxed'] and not report['policy']['failed_gate']['boxed']
assert report['policy']['enabled']['boxed'] and len(report['policy']['enabled']['factory_calls'])==1
# CUDA Graph backward被禁用时，仍通知既有tree manager；不捕获GPU。
g=graph(['fixture reject'],backward=True);g.inputs_to_check=[];events=[]
manager=types.SimpleNamespace(set_to_running_backward=lambda:events.append('backward'))
def backward_model(args):events.append('model');return model(args)
g.current_callable=backward_model;flag=u.BoxedBool(True)
with config.patch({'triton.cudagraph_trees':True}),patch.object(trees,'get_manager',return_value=manager):cc.cudagraph_post_compile([],g,flag)
g.current_callable([torch.ones(1)])
assert events==['backward','model'] and not flag
report['policy']['backward_notice']={'events':events,'boxed':bool(flag),'manager_fixture':True}
# 真实延迟cudagraphify包装，只替换捕获factory；第二次调用不再初始化。
for tree_mode in [False,True]:
    calls=[]
    def capture(fn,inputs,static,*args,**kwargs):
        calls.append({'static':list(static),'kwargs':sorted(kwargs)});return fn
    target=(trees,'cudagraphify_impl') if tree_mode else (cf,'cudagraphify_impl')
    with config.patch({'triton.cudagraph_trees':tree_mode}),patch.object(*target,side_effect=capture),patch.object(cf.dynamo_utils,'preserve_rng_state',contextlib.nullcontext):
        wrapped=cf.cudagraphify(model,[0],device_index=2,stack_traces=[],is_backward=False,is_inference=True)
        assert not calls
        a=wrapped([torch.tensor([3.])])[0];b=wrapped([torch.tensor([8.])])[0]
        assert a.item()==4 and b.item()==9 and len(calls)==1
    report['deferred_protocol']['trees_'+str(tree_mode)]={'factory_calls':len(calls),'outputs':[a.item(),b.item()],'factory_kwargs':calls[0]['kwargs'],'capture_fixture':True}
# 初次捕获factory异常传播，不自动走普通model；下一次调用才重试。
calls=[]
def flaky(fn,inputs,static):
    calls.append(1)
    if len(calls)==1:raise ValueError('fixture capture failed')
    return fn
with config.patch({'triton.cudagraph_trees':False}),patch.object(cf,'cudagraphify_impl',side_effect=flaky),patch.object(cf.dynamo_utils,'preserve_rng_state',contextlib.nullcontext):
    wrapped=cf.cudagraphify(model,device_index=2,stack_traces=[],is_backward=False,is_inference=True)
    try:wrapped([torch.ones(1)])
    except ValueError as e:error=str(e)
    else:raise AssertionError('capture error swallowed')
    assert wrapped([torch.ones(1)])[0].item()==2 and len(calls)==2
report['deferred_protocol']['capture_error']={'error':error,'factory_attempts':len(calls),'capture_fixture':True}
# Tree延迟整型键缓存：只用CPU张量和普通int，替换实际manager/capture。
keys=[]
def int_model(args):
    n,x=args;args.clear();return (x+n,)
def tree_factory(fn,inputs,static,*a,**kw):
    keys.append(inputs[0]);return fn,fn(inputs)
with patch.object(trees,'cudagraphify',side_effect=tree_factory):
    deferred=trees.cudagraphify_impl(int_model,[7,torch.ones(1)],[])
    vals=[deferred([n,torch.ones(1)])[0].item() for n in [7,7,8]]
assert keys==[7,8] and vals==[8.,8.,9.]
report['deferred_protocol']['tree_integer_keys']={'factory_keys':keys,'outputs':vals,'manager_fixture':True}
# 同一mutation在trees中可因static参数豁免；legacy仍拒绝。
gm=make_fx(lambda x:(x.add_(1),))(torch.ones(2));desc=types.SimpleNamespace(mutated_input_idxs={0},mutated_inputs={'x'})
with config.patch({'triton.cudagraph_trees':True}):
    static_reason=check_for_mutation_ignore_cuda_graph_managed_tensor(gm,desc,[0]);dynamic_reason=check_for_mutation_ignore_cuda_graph_managed_tensor(gm,desc,[])
with config.patch({'triton.cudagraph_trees':False}):legacy_reason=check_for_mutation_ignore_cuda_graph_managed_tensor(gm,desc,[0])
assert static_reason is None and dynamic_reason and legacy_reason
report['policy']['mutation_gate']={'trees_static_allowed':True,'trees_dynamic_rejected':True,'legacy_static_rejected':True,'metadata_only':True}
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
