"""真实 CPU 编译 Future 在 wait 后成为可调用入口；不测并行收益。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-async-loading')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import CodeCacheFuture
source='''extern "C" void kernel(float* ptr) { ptr[0] = ptr[0] + 7.0f; }'''
report={}
with config.patch({'compile_threads':2}):
    compiler=AsyncCompile()
    pending=compiler.cpp_pybinding(['float*'],source)
    report['before_type']=type(pending).__name__
    report['before_is_future']=isinstance(pending,CodeCacheFuture)
    scope={'learning_kernel':pending}
    compiler.wait(scope)
    ready=scope['learning_kernel']
    report['after_type']=type(ready).__name__
    report['after_is_future']=isinstance(ready,CodeCacheFuture)
    report['callable_after_wait']=callable(ready)
    value=torch.tensor([5.])
    ready(value)
    assert value.item()==12.
    report['actual']=value.tolist()
    assert report['before_is_future'] and not report['after_is_future']
(ROOT/'experiments/async_loading.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(ROOT/'experiments/async_loading.cpp').write_text(source+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
