"""真实后端注册消费与 PrivateUse1 分支；CPU 完整执行另列，私有设备不执行 kernel。"""
from pathlib import Path
import os,json,sys,subprocess,types
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/backend-registration';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-backend-registration')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'

def worker(case):
    import torch
    from torch._inductor.codegen import common as c
    from torch._inductor import config
    from torch._inductor.graph import GraphLowering
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.virtualized import V
    events=[]
    class Scheduling:
        def __init__(self,scheduler):self.scheduler=scheduler;events.append({'kind':'scheduling','probe':scheduler is None})
        def get_backend_features(self,device):return {c.BackendFeature.INPLACE_BUFFERS}
    class PythonWrapper:
        def __init__(self):events.append({'kind':'python_wrapper'})
    class CppWrapper:
        def __init__(self):events.append({'kind':'cpp_wrapper'})
    class Ops(c.DeviceOpOverrides):
        def import_get_raw_stream_as(self,name):return f'from torch import lesson_stream as {name}'
        def set_device(self,index):return f'lesson_device.set_device({index})'
        def synchronize(self):return 'lesson_device.synchronize()'
        def device_guard(self,index):return f'lesson_device.guard({index})'
    if case.startswith('private_'):
        initial=torch._C._get_privateuse1_backend_name()
        assert initial=='privateuseone',initial
        if case!='private_default':torch.utils.rename_privateuse1_backend('lesson_device')
        device_name=torch._C._get_privateuse1_backend_name()
        module=types.ModuleType('lesson_device')
        if case not in ['private_no_module','private_late_module','private_default']:
            module.Scheduling=Scheduling
            if case!='private_partial_module':module.WrapperCodeGen=PythonWrapper
            if case not in ['private_partial_module','private_false_cpp']:module.CppWrapperCodeGen=CppWrapper
            elif case=='private_false_cpp':module.CppWrapperCodeGen=None
            torch._register_device_module(device_name,module)
        if case=='private_explicit':c.register_backend_for_device(device_name,Scheduling,PythonWrapper)
        c.init_backend_registration()
        before=c.get_scheduling_for_device(device_name)
        result={'initial_private_name':initial,'device_name':device_name,'registered':before is not None,'cache_info':list(c.init_backend_registration.cache_info()),'scope':'real registration/selectors only; no new-device allocation, kernel or runtime'}
        if case=='private_late_module':
            module.Scheduling=Scheduling;module.WrapperCodeGen=PythonWrapper;module.CppWrapperCodeGen=CppWrapper
            torch._register_device_module(device_name,module);c.init_backend_registration()
            result['after_second_init']=c.get_scheduling_for_device(device_name) is not None
            c.init_backend_registration.cache_clear();c.init_backend_registration()
            result['after_cache_clear_init']=c.get_scheduling_for_device(device_name) is not None
        if case in ['private_complete','private_explicit']:
            result.update({'python_selected':c.get_wrapper_codegen_for_device(device_name).__name__,'cpp_selected':c.get_wrapper_codegen_for_device(device_name,True).__name__,'features':[v.name for v in c.get_backend_features(torch.device(device_name,0))]})
        if case=='private_complete':
            ops=Ops();c.register_device_op_overrides(device_name,ops)
            receiver=types.SimpleNamespace(device_types={'cpu',device_name},cpp_wrapper=False,const_module=None)
            GraphLowering.init_wrapper_code(receiver)
            assert isinstance(receiver.wrapper_code,PythonWrapper) and receiver.device_ops is ops
            result['graph_wrapper_consumer']={'wrapper':type(receiver.wrapper_code).__name__,'ops_same_object':receiver.device_ops is ops,'scope':'actual init_wrapper_code method with minimal graph-state fixture'}
            class BackendReceiver:
                def __init__(self):self.backends={}
                def create_backend(self,device):return Scheduler.create_backend(self,device)
            seen=[];graph_context=types.SimpleNamespace(add_device_info=lambda dev:seen.append(str(dev)))
            scheduler=BackendReceiver()
            with V.set_graph_handler(graph_context):
                first=Scheduler.get_backend(scheduler,torch.device(device_name,0));again=Scheduler.get_backend(scheduler,torch.device(device_name,0));second=Scheduler.get_backend(scheduler,torch.device(device_name,1))
            assert first is again and first is not second
            result['scheduler_consumer']={'same_device_cached':first is again,'different_indices_distinct':first is not second,'add_device_info':seen,'scope':'actual get_backend/create_backend with graph-state fixture; backend objects have no kernel codegen'}
        assert result['registered']==(case in ['private_complete','private_explicit'])
        if case=='private_late_module':assert not result['after_second_init'] and result['after_cache_clear_init']
        return result
    if case=='selectors':
        c.register_backend_for_device('not_a_torch_device',Scheduling,PythonWrapper)
        accepted=c.get_scheduling_for_device('not_a_torch_device') is Scheduling
        try:torch.device('not_a_torch_device')
        except RuntimeError as e:invalid=type(e).__name__
        else:raise AssertionError('registry string should not create a device type')
        c.register_backend_for_device('cpu',Scheduling,PythonWrapper)
        c.init_backend_registration();kept=c.get_scheduling_for_device('cpu') is Scheduling
        features=[v.name for v in c.get_backend_features('cpu')]
        first=c.get_scheduling_for_device('cpu')
        class Replacement(Scheduling):pass
        c.register_backend_for_device('cpu',Replacement,CppWrapper)
        return {'arbitrary_key_accepted':accepted,'torch_device_error':invalid,'pre_registration_preserved':kept,'features':features,'replacement_overwrites':c.get_scheduling_for_device('cpu') is Replacement,'python_wrapper_selected':c.get_wrapper_codegen_for_device('cpu').__name__,'missing_device':c.get_scheduling_for_device('absent'),'omitted_cpp_wrapper':c.get_wrapper_codegen_for_device('cpu',True).__name__,'feature_probe_with_none':any(e.get('probe') for e in events)}
    if case=='device_ops':
        ops=Ops();c.register_device_op_overrides('lesson',ops)
        missing=c.get_device_op_overrides('cuda')
        only=list(c.device_op_overrides_dict)
        # 主动导入实际模块后，已登记的 CUDA override 才能被查到。
        from torch._inductor.codegen.cuda import device_op_overrides as cuda_ops
        from torch._inductor.codegen.xpu import device_op_overrides as xpu_ops
        actual=c.get_device_op_overrides('cuda')
        try:c.DeviceOpOverrides().set_device(0)
        except NotImplementedError as e:base_error=type(e).__name__
        else:raise AssertionError('base method should not implement device code')
        return {'unrelated_registration_prevents_lazy_builtin_import':missing is None,'before_explicit_import':only,'cuda_after_explicit_import':type(actual).__name__,'emitted_guard':actual.device_guard(2),'emitted_stream_import':actual.import_get_raw_stream_as('stream_alias'),'custom_same_object':c.get_device_op_overrides('lesson') is ops,'base_error':base_error,'scope':'generated source strings only; no set_device or synchronization executed'}
    if case=='cpu_compile':
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._inductor.codegen.cpp import CppScheduling
        from torch._inductor.codegen.wrapper import WrapperCodeGen
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.compile_fx import compile_fx_inner
        class LessonScheduling(CppScheduling):
            def __init__(self,scheduler):events.append({'kind':'scheduling','probe':scheduler is None});super().__init__(scheduler)
            def group_fn(self,sizes):events.append({'kind':'group','sizes':str(sizes)});return super().group_fn(sizes)
            def codegen_node(self,node):events.append({'kind':'codegen_node','node_type':type(node).__name__});return super().codegen_node(node)
            def flush(self):events.append({'kind':'flush'});return super().flush()
        class LessonWrapper(WrapperCodeGen):
            def __init__(self):events.append({'kind':'wrapper'});super().__init__()
            def generate_kernel_call(self,*args,**kwargs):events.append({'kind':'kernel_call','name':str(args[0]) if args else kwargs.get('kernel_name')});return super().generate_kernel_call(*args,**kwargs)
        c.register_backend_for_device('cpu',LessonScheduling,LessonWrapper,CppWrapperCpu)
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)-3;y=torch.ones(2,3)
        fn=lambda x,y:(torch.relu(x+y)*2,)
        gm=make_fx(fn)(x,y);original=GraphLowering.save_output_code
        GraphLowering.save_output_code=staticmethod(lambda code:(OUT/'cpu_output.py').write_text(code))
        try:
            with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):compiled=compile_fx_inner(gm,[x.clone(),y.clone()],is_inference=True)
        finally:GraphLowering.save_output_code=staticmethod(original)
        ai=[x.clone(),y.clone()];actual=compiled(list(ai));torch.testing.assert_close(actual,fn(x,y));torch.testing.assert_close(ai,[x,y])
        assert not any(actual[0].untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in ai)
        kinds=[e['kind'] for e in events]
        assert all(k in kinds for k in ['wrapper','scheduling','group','codegen_node','flush','kernel_call'])
        assert c.get_scheduling_for_device('cpu') is LessonScheduling
        return {'registered_classes_preserved':True,'events':events,'output':actual[0].tolist(),'input_unchanged':True,'output_not_aliasing_inputs':True,'scope':'real complete CPU compilation delegating to existing CppScheduling; not new hardware execution'}
    raise AssertionError(case)

if __name__=='__main__':
    if len(sys.argv)>1:
        print(json.dumps(worker(sys.argv[1]),ensure_ascii=False));sys.exit()
    report={}
    cases=['private_default','private_no_module','private_partial_module','private_false_cpp','private_complete','private_explicit','private_late_module','selectors','device_ops','cpu_compile']
    for case in cases:
        process=subprocess.run([sys.executable,__file__,case],text=True,capture_output=True,check=False)
        (OUT/f'{case}_stderr.txt').write_text(process.stderr)
        if process.returncode:raise RuntimeError(f'{case}: {process.stderr}')
        report[case]=json.loads(process.stdout)
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
