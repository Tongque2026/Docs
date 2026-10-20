"""记录未经上游 decomposition 的 raw t 图在单图入口的真实失败。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-entry-boundary')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
gm=make_fx(lambda x:(x.t(),))(x)
report={'graph':str(gm.graph),'scope':'raw make_fx single-graph entry; not complete torch.compile pipeline'}
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'implicit_fallbacks':True}):compile_fx_inner(gm,[x],is_inference=True)
except AssertionError as e:
    assert 'both a fallback and a decomp for same op: aten.t.default' in str(e)
    report.update(error=type(e).__name__,message=str(e))
else:raise AssertionError('expected this checkout to reject raw aten.t at single-graph entry')
(ROOT/'experiments/entry_boundary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
