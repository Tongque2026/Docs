"""真实 CPU 编译、线程/进程任务，加明确标注的 Future/Triton 描述协议实验。"""
import json, os, sys, tempfile, threading
from pathlib import Path
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCH_WARM_POOL']='0'
os.environ['TORCHINDUCTOR_CACHE_DIR']=tempfile.mkdtemp(prefix='tasks-',dir=ROOT/'experiments')
import torch
from torch._inductor import config
import torch._inductor.async_compile as ac
from torch._inductor.codecache import CodeCacheFuture, TritonFuture
from torch._inductor.runtime import compile_tasks as ct
from compile_task_helpers import identity, failing
report={'cpu':{},'protocol':{}}
compiler=ac.AsyncCompile()
main=identity(); report['main']=main
source='extern "C" void kernel(float* ptr) { ptr[0] += 7.0f; }'
for threads in (1,2):
    with config.patch({'compile_threads':threads,'disable_progress':True}):
        task=compiler.submit(identity)
        task_result=task if threads==1 else task.result(timeout=30)
        submissions=[]
        original_submit=compiler.submit
        def submit_compile(task):
            submissions.append(type(task).__name__)
            return original_submit(task)
        with patch.object(compiler,'submit',side_effect=submit_compile):
            pending=compiler.cpp_pybinding(['float*'],source+'\n// compile_threads='+str(threads))
        assert bool(submissions)==(threads==2)
        before=type(pending).__name__
        scope={'kernel':pending}; assert compiler.wait(scope) is None
        x=torch.tensor([5.]);scope['kernel'](x)
        assert x.item()==12.
        assert isinstance(pending,CodeCacheFuture)==(threads==2)
        assert task_result['pid']==main['pid']
        assert (task_result['thread']==main['thread'])==(threads==1)
        report['cpu'][str(threads)]={'compile_submissions':len(submissions),'before':before,'after':type(scope['kernel']).__name__,'output':x.tolist(),'task':task_result}

def ready(value=None,error=None):
    f=Future()
    if error is None:f.set_result(value)
    else:f.set_exception(error)
    return f
with config.patch({'compile_threads':2,'disable_progress':True}):
    nested=ready('nested');scope={'first':ready(17),'nested':[nested],'literal':23}
    compiler.wait(scope)
    assert scope['first']==17 and scope['nested'][0] is nested
    report['protocol']['shallow_scope']={'first':17,'nested_preserved':True,'literal':23}
    bad=ready(error=ValueError('explicit failure'));later=ready(99)
    scope={'first':ready(17),'bad':bad,'later':later}
    try:compiler.wait(scope)
    except ValueError as e:report['protocol']['partial_failure']={'error':type(e).__name__,'first':scope['first'],'bad_same':scope['bad'] is bad,'later_same':scope['later'] is later}
    else:raise AssertionError('failure swallowed')
    try:compiler.wait({'broken':ready(error=BrokenProcessPool('synthetic pool failure'))})
    except RuntimeError as e:
        assert 'TORCHINDUCTOR_COMPILE_THREADS=1' in str(e)
        report['protocol']['broken_pool']={'error':type(e).__name__,'cause':type(e.__cause__).__name__}
with config.patch({'compile_threads':1,'disable_progress':True}):
    f=ready(8);scope={'f':f};compiler.wait(scope);assert scope['f'] is f
    report['protocol']['threads1_wait_preserves_injected_future']=True
# 真正 subprocess sidecar + fork workers，任务仅返回 Python 标量。
with config.patch({'compile_threads':2,'worker_start_method':'subprocess','disable_progress':True}):
    pool=compiler.process_pool()
    try:
        assert pool.ready_future.result(timeout=60)=='ready'
        assert compiler._use_process_pool()
        worker=pool.submit(identity).result(timeout=30)
        assert worker['pid']!=main['pid']
        scope={'job':pool.submit(identity)};compiler.wait(scope)
        assert scope['job']['pid']!=main['pid']
        try:pool.submit(failing).result(timeout=30)
        except Exception as e:
            assert type(e).__name__=='SubprocException' and 'learning compile task failed' in str(e)
            failure=type(e).__name__
        else:raise AssertionError('worker error swallowed')
        report['process']={'worker':worker,'ready':True,'wait_result':scope['job'],'ordinary_job_error':failure,'device_compilation':False}
    finally:ac.shutdown_compile_workers()
# 以下 kernel、池和 loader 是 fixture；运行真实 dispatch、worker helper、TritonFuture。
class Kernel:
    def __init__(self):self.flags=[];self._reload_in_subproc=lambda:self
    def precompile(self,**kw):self.flags.append(kw)
class Pool:
    def __init__(self):self.calls=[]
    def submit(self,fn,*args):self.calls.append((fn,args));return ready()
for use_pool in (False,True):
    kernel=Kernel();pool=Pool()
    with patch.object(ac.TritonCodeCache,'load',return_value=kernel),patch.object(ac,'_set_triton_ptxas_path'),patch.object(ac.AsyncCompile,'_use_process_pool',return_value=use_pool),patch.object(ac.AsyncCompile,'process_pool',return_value=pool),patch.dict(os.environ,{'TRITON_CACHE_DIR':'fixture-cache'}):
        result=compiler.triton('fixture','not executable source',device_str='new-device-fixture')
        if use_pool:
            assert isinstance(result,TritonFuture) and kernel.flags==[]
            fn,args=pool.calls[0]
            assert fn is ac._worker_compile_triton
            assert set(args[1])=={'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'}
            # helper 在本进程受控运行；restore env，未在 GPU worker 里编译。
            with patch.object(ct,'_set_triton_ptxas_path'),patch.dict(os.environ):fn(*args)
            assert kernel.flags==[{'warm_cache_only':True}]
            assert result.result() is kernel and result.result() is kernel
            assert kernel.flags==[{'warm_cache_only':True},{}]
        else:
            assert result is kernel and kernel.flags==[{}]
        report['protocol']['triton_'+str(use_pool)]={'precompile_flags':kernel.flags,'device_str_observed_by_kernel':False,'fixtures_only':True}
ac.AsyncCompile.pool().shutdown(wait=True)
ac.AsyncCompile.pool.cache_clear()
out=ROOT/'experiments/compile-tasks.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(out.read_text())
