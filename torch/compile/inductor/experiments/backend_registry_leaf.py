"""注册叶子补充分支：配置延迟选择、能力探测、字符串键、导入缓存、初始化异常。"""
from pathlib import Path
import os,sys,json,subprocess,types
R=Path(__file__).resolve().parents[1];O=R/'experiments/backend-registry-leaf';O.mkdir(exist_ok=True)
os.environ['TORCH_WARM_POOL']='0'
def worker(case):
    import torch
    from unittest.mock import patch
    from torch._inductor.codegen import common as c
    from torch._inductor import config
    class Wrapper:pass
    class CppWrapper:pass
    class Probe:
        events=[]
        features={c.BackendFeature.INPLACE_BUFFERS:None}
        def __init__(self,scheduler):self.events.append({'scheduler_none':scheduler is None})
        def get_backend_features(self,device):self.events.append({'device':str(device)});return self.features
    c.device_codegens.clear();c.init_backend_registration.cache_clear()
    if case=='dynamic_defaults':
        assert c.get_scheduling_for_device('cpu') is None and not c.device_codegens
        c.init_backend_registration();factory=c.get_scheduling_for_device('cpu')
        with config.patch({'cpu_backend':'cpp'}):cpp=type(factory(None)).__name__;a=c.get_backend_features('cpu')
        with config.patch({'cpu_backend':'halide','halide.scan_kernels':False}):halide=type(factory(None)).__name__;b=c.get_backend_features('cpu')
        with config.patch({'cpu_backend':'not-a-backend'}):
            try:factory(None)
            except KeyError:invalid='KeyError'
            else:raise AssertionError('config should not select implementation')
        assert factory is c.get_scheduling_for_device('cpu') and cpp=='CppScheduling' and halide=='HalideScheduling'
        return {'getter_does_not_init':True,'same_factory':True,'cpp_kind':cpp,'halide_kind':halide,'cpp_features':sorted(x.name for x in a),'halide_features':sorted(x.name for x in b),'invalid_config':invalid,'xpu_cpp_slot_is_none_type':c.get_wrapper_codegen_for_device('xpu',True) is type(None),'no_kernel_compilation':True}
    if case=='feature_probes':
        c.register_backend_for_device('cpu',Probe,Wrapper)
        first=c.get_backend_features('cpu')
        yes=c.has_backend_feature('cpu',c.BackendFeature.INPLACE_BUFFERS);no=c.has_backend_feature('cpu',c.BackendFeature.SCAN)
        probes=sum('scheduler_none' in e for e in Probe.events);assert probes==3 and first is Probe.features and yes and not no
        object_features=c.get_backend_features(torch.device('cpu:0'))
        try:c.get_backend_features('cpu:0')
        except TypeError as e:string_error=type(e).__name__
        else:raise AssertionError('string raw key not normalized')
        class Indexed(Probe):features={c.BackendFeature.SCAN:None}
        c.register_backend_for_device('cpu:0',Indexed,Wrapper)
        a=c.get_backend_features('cpu:0');b=c.get_backend_features(torch.device('cpu:0'))
        assert c.BackendFeature.SCAN in a and c.BackendFeature.INPLACE_BUFFERS in b
        try:c.has_backend_feature('cpu','INPLACE_BUFFERS')
        except AssertionError:bad_feature='AssertionError'
        else:raise AssertionError('must pass Enum')
        try:c.get_backend_features('meta')
        except TypeError:missing='TypeError'
        else:raise AssertionError('missing scheduling should fail')
        return {'initial_probe_instances':probes,'returns_same_feature_object':True,'membership_true':yes,'membership_false':no,'cpu_index_string_error':string_error,'string_exact_key_features':sorted(x.name for x in a),'device_object_type_features':sorted(x.name for x in b),'bad_feature':bad_feature,'missing_scheduling':missing,'events':Probe.events,'metadata_only':True}
    if case=='ops_import_cache':
        c.device_op_overrides_dict.clear()
        first=c.get_device_op_overrides('cuda');assert first is not None
        keys=sorted(c.device_op_overrides_dict)
        c.device_op_overrides_dict.clear();second=c.get_device_op_overrides('cuda')
        assert second is None and not c.device_op_overrides_dict
        custom=object();c.register_device_op_overrides('cuda',custom)
        assert c.get_device_op_overrides('cuda') is custom
        replacement=object();c.register_device_op_overrides('cuda',replacement)
        assert c.get_device_op_overrides('cuda') is replacement
        errors=[]
        for name,args in [('import_get_raw_stream_as',('s',)),('set_device',(0,)),('synchronize',()),('device_guard',(0,))]:
            try:getattr(c.DeviceOpOverrides(),name)(*args)
            except NotImplementedError:errors.append(name)
        assert len(errors)==4
        return {'fresh_import_keys':keys,'clear_after_import_returns_none':second is None,'explicit_overwrite_identity':True,'base_methods_reject':errors,'generated_strings_only':True}
    if case=='private_exception':
        torch.utils.rename_privateuse1_backend('leaf_device')
        module=types.ModuleType('leaf_device');module.Scheduling=Probe;module.WrapperCodeGen=Wrapper;module.CppWrapperCodeGen=CppWrapper
        torch._register_device_module('leaf_device',module)
        import torch.utils.backend_registration as br
        with patch.object(br,'_get_custom_mod_func',side_effect=ValueError('fixture custom getter failure')):
            try:c.init_backend_registration()
            except ValueError:error='ValueError'
            else:raise AssertionError('non RuntimeError must propagate')
        cache_after_error=list(c.init_backend_registration.cache_info())
        defaults_after_error=sorted(c.device_codegens)
        assert cache_after_error[-1]==0 and c.get_scheduling_for_device('cpu') is not None
        c.init_backend_registration()
        assert c.get_scheduling_for_device('leaf_device') is Probe
        return {'getter_fixture_error':error,'failed_init_cached_entries':cache_after_error[-1],'defaults_remain_after_error':defaults_after_error,'retry_registered':True,'cache_after_retry':list(c.init_backend_registration.cache_info()),'no_device_execution':True}
    if case=='private_false_value':
        torch.utils.rename_privateuse1_backend('false_device')
        module=types.ModuleType('false_device');module.Scheduling=Probe;module.WrapperCodeGen=Wrapper;module.CppWrapperCodeGen=False
        torch._register_device_module('false_device',module)
        import torch.utils.backend_registration as br
        original=br._get_custom_mod_func;events=[]
        def getter(name):
            try:value=original(name)
            except RuntimeError:
                events.append({'name':name,'error':'RuntimeError'});raise
            events.append({'name':name,'truth':bool(value)});return value
        with patch.object(br,'_get_custom_mod_func',side_effect=getter):c.init_backend_registration()
        false_events=list(events);assert c.get_scheduling_for_device('false_device') is None
        assert false_events[-1]=={'name':'CppWrapperCodeGen','truth':False}
        module.CppWrapperCodeGen=None;events.clear();c.init_backend_registration.cache_clear()
        with patch.object(br,'_get_custom_mod_func',side_effect=getter):c.init_backend_registration()
        assert events[-1]=={'name':'CppWrapperCodeGen','error':'RuntimeError'}
        assert c.get_scheduling_for_device('false_device') is None
        return {'false_value_events':false_events,'none_value_events':events,'both_unregistered':True,'no_device_execution':True}
    if case=='raw_slots':
        c.register_backend_for_device('cpu',None,Wrapper)
        c.init_backend_registration();assert c.get_scheduling_for_device('cpu') is not None
        c.register_backend_for_device('anything',object(),None,None)
        assert c.get_wrapper_codegen_for_device('anything') is None and c.get_wrapper_codegen_for_device('anything',True) is None
        omitted='omitted';c.register_backend_for_device(omitted,Probe,Wrapper)
        assert c.get_wrapper_codegen_for_device(omitted,True) is type(None)
        assert c.get_scheduling_for_device('absent') is None and c.get_wrapper_codegen_for_device('absent') is None
        c.register_backend_for_device('arbitrary',Probe,Wrapper)
        try:c.get_backend_features('arbitrary')
        except RuntimeError:error='RuntimeError'
        else:raise AssertionError('torch.device must recognize type')
        return {'none_scheduling_replaced_by_default':True,'invalid_implementation_slots_stored':True,'explicit_none_slot':True,'omitted_cpp_slot_is_none_type':True,'missing_getters_none':True,'arbitrary_feature_query_error':error}
    raise AssertionError(case)
if len(sys.argv)>1:
    print(json.dumps(worker(sys.argv[1]),ensure_ascii=False));raise SystemExit
report={}
for case in ['dynamic_defaults','feature_probes','ops_import_cache','private_exception','private_false_value','raw_slots']:
    cp=subprocess.run([sys.executable,__file__,case],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH='/work/pytorch'),timeout=60)
    (O/(case+'_stderr.txt')).write_text(cp.stderr)
    if cp.returncode:raise RuntimeError(case+'\n'+cp.stderr)
    report[case]=json.loads(cp.stdout)
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
