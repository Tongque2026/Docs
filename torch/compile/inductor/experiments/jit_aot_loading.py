"""真实JIT模块、PyCodeCache协议和独立进程AOT构建/运行尝试。"""
import json,os,sys,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'experiments/jit-aot-loading';O.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=tempfile.mkdtemp(prefix='cache-',dir=O)
os.environ['TORCH_WARM_POOL']='0';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from unittest.mock import patch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx
import torch._inductor.codecache as cc
x=torch.arange(6,dtype=torch.float32).reshape(2,3);w=torch.tensor([10.,20.,30.])
if '--aot-child' in sys.argv:
    report={}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('weight',w)
        def forward(self,x):return (x+self.weight,)
    gm=torch.export.export(Model(),(x,)).module()
    try:
        original_compile=cc.AotCodeCompiler.compile
        def capture_aot(graph,source,*args,**kwargs):
            (O/'aot_output.cpp').write_text(source)
            return original_compile(graph,source,*args,**kwargs)
        with patch.object(cc.AotCodeCompiler,'compile',side_effect=capture_aot):
            path=torch._inductor.aot_compile(gm,(x,),options={'compile_threads':1,'cpp.simdlen':1,'aot_inductor.output_path':str(O/'model.so'),'aot_inductor.package':False,'abi_compatible':True,'always_keep_tensor_constants':True})
        report={'compiled':True,'path':path,'exists':Path(path).exists()}
        runner=torch._C._aoti.AOTIModelContainerRunnerCpu(path,1)
        before=x.clone();actual=runner.run([x]);torch.testing.assert_close(actual[0],x+w)
        assert torch.equal(x,before)
        assert actual[0].untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
        report.update(constant_names_to_fqns=runner.get_constant_names_to_original_fqns(),loaded=True,output=actual[0].tolist(),input_preserved=True,output_aliases_input=False,call_spec=list(runner.get_call_spec()))
    except Exception as e:
        report.update(error=type(e).__name__,message=str(e));print(str(e),file=sys.stderr)
    (O/'aot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False));raise SystemExit
report={'jit':{},'module_protocol':{}}
events=[];orig_load=cc.PyCodeCache.load_by_key_path;orig_reload=cc._reload_python_module
captured={}
def reload_module(key,path):
    mod=orig_reload(key,path)
    events.append({'phase':'module_executed','key':key,'constant_names_present':[k for k in captured.get('expected_attrs',[]) if k in mod.__dict__],'constant_placeholders':{k:mod.__dict__.get(k) is None for k in captured.get('expected_attrs',[])}})
    return mod

def load(key,path,linemap=None,attrs=None):
    captured['expected_attrs']=list(attrs or {})
    events.append({'phase':'load_request','key':key,'attrs':list(attrs or {}),'linemap_count':len(linemap or [])})
    mod=orig_load(key,path,linemap,attrs)
    events.append({'phase':'load_return','attrs_present':all(getattr(mod,k) is v for k,v in (attrs or {}).items()),'callable':callable(mod.call)})
    captured['mod']=mod
    return mod
with patch.object(cc.PyCodeCache,'load_by_key_path',side_effect=load),patch.object(cc,'_reload_python_module',side_effect=reload_module),patch.object(GraphLowering,'save_output_code',side_effect=lambda code:(O/'jit_output.py').write_text(code)),config.patch({'force_disable_caches':True,'fx_graph_cache':False,'cpp.simdlen':1,'triton.cudagraphs':False,'always_keep_tensor_constants':True}):
    gm=make_fx(lambda x:(torch.relu(x+w),),decomposition_table=select_decomp_table())(x)
    compiled=compile_fx_inner(gm,[x],is_inference=True)
    before=x.clone();wb=w.clone();args=[x];actual=compiled(args)
    torch.testing.assert_close(actual[0],torch.relu(x+w));assert torch.equal(x,before) and torch.equal(w,wb)
    report['jit']={'events':events,'attrs':list(captured['expected_attrs']),'module_file':captured['mod'].__file__,'key':captured['mod'].key,'module_has_call':callable(captured['mod'].call),'output':actual[0].tolist(),'input_preserved':True,'constant_preserved':True,'args_consumed':args==[],'output_aliases_input':actual[0].untyped_storage().data_ptr()==x.untyped_storage().data_ptr()}
# 以下是小型Python模块的真实加载协议，不冒充生成器产生的kernel。
source="seen_constant_at_import = 'constant0' in globals()\ndef call(args):\n    return (args[0] + constant0,)\n"
key,path=cc.PyCodeCache.write(source)
trace='  File "learning_model.py", line 7, in forward\n'
a=torch.tensor([10.]);b=torch.tensor([20.]);line=[(2,trace)]
m1=cc.PyCodeCache.load_by_key_path(key,path,line,{'constant0':a})
m2=cc.PyCodeCache.load_by_key_path(key,path,[(2,'different')],{'constant0':b})
assert m1 is m2 and m2.constant0 is a and not m1.seen_constant_at_import
assert not hasattr(m1,'_reload_in_subproc') and cc.PyCodeCache.linemaps[path]==list(zip(*line))
assert m1.call([torch.tensor([1.])])[0].item()==11
frames=cc.PyCodeCache.stack_frames_for_code(path,2)
assert frames==[{'filename':'learning_model.py','line':7,'name':'forward'}]
report['module_protocol']['attrs_cache_hit']={'same_module':True,'original_constant_retained':True,'seen_constant_during_exec':False,'has_reload_callback':False,'value':11,'frames':frames}
plain=cc.PyCodeCache.load('value = 3\n')
assert hasattr(plain,'_reload_in_subproc')
cc.PyCodeCache.cache.pop(plain.key)
reloaded=plain._reload_in_subproc();assert reloaded is not plain and reloaded.value==3
report['module_protocol']['plain_reload']={'new_module':True,'value':3,'same_path':plain.__file__==reloaded.__file__}
for label,source in [('syntax','def broken(:\n'),('exec',"raise ValueError('module execution failed')\n")]:
    try:cc.PyCodeCache.load(source)
    except Exception as e:report['module_protocol'][label]={'error':type(e).__name__,'message':str(e)}
    else:raise AssertionError(label)
assert report['module_protocol']['syntax']['error']=='RuntimeError' and report['module_protocol']['exec']['error']=='ValueError'
# 实际aot_load入口的设备分流：runner是fixture，不调用GPU。
import torch._export as ex
import torch.utils._pytree as pytree
in_spec=pytree.tree_flatten(((torch.ones(1),),{}))[1];out_spec=pytree.tree_flatten((torch.ones(1),))[1]
class Runner:
    def __init__(self,*args):self.args=args
    def get_call_spec(self):return [pytree.treespec_dumps(in_spec),pytree.treespec_dumps(out_spec)]
    def run(self,inputs):return [inputs[0]+2]
with patch.object(torch._C._aoti,'AOTIModelContainerRunnerCpu',Runner),patch.object(torch._C._aoti,'AOTIModelContainerRunnerCuda',Runner,create=True):
    cpu=ex.aot_load('fixture.so','cpu');gpu=ex.aot_load('fixture.so','cuda:1')
    assert cpu(torch.tensor([3.]))[0].item()==gpu(torch.tensor([3.]))[0].item()==5
    try:ex.aot_load('fixture.so','privateuseone:0')
    except RuntimeError as e:unsupported=str(e)
    else:raise AssertionError('unknown device accepted')
report['module_protocol']['aot_loader_fixture']={'cpu_value':5,'cuda_value':5,'unsupported_error':unsupported,'fixtures_only':True}
(O/'aot.json').unlink(missing_ok=True)
with (O/'aot_stdout.txt').open('w') as out,(O/'aot_stderr.txt').open('w') as err:
    cp=subprocess.run([sys.executable,__file__,'--aot-child'],stdout=out,stderr=err,timeout=180,env=dict(os.environ,PYTHONPATH='/work/pytorch'))
report['aot_process_exit']=cp.returncode
report['aot']=json.loads((O/'aot.json').read_text()) if (O/'aot.json').exists() else {'compiled':False,'process_failure':cp.returncode}
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
