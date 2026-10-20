"""独立进程验证 FX 图缓存 miss/hit/bypass，直接计数 Lowering 与 post_compile。"""
import os,sys,json,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/cache-lifecycle';OUT.mkdir(exist_ok=True)
if len(sys.argv)==1:
    parent=ROOT/'experiments/cache-fxgraph-proof';parent.mkdir(exist_ok=True)
    cache=tempfile.mkdtemp(prefix='run-',dir=parent)
    rows={}
    for label in ['cold','warm','new_shape','new_dtype','bypass']:
        child=subprocess.run([sys.executable,__file__,'worker',label,cache],text=True,capture_output=True)
        if child.returncode:
            print(child.stdout);print(child.stderr,file=sys.stderr);raise SystemExit(child.returncode)
        rows[label]=json.loads((OUT/f'{label}.json').read_text())
    assert rows['cold']['calls']['lowering']==1 and rows['cold']['counters'].get('fxgraph_cache_miss')==1
    assert rows['warm']['calls']['lowering']==0 and rows['warm']['counters'].get('fxgraph_cache_hit')==1
    assert rows['warm']['cache_key']==rows['cold']['cache_key']
    assert rows['new_shape']['calls']['lowering']==0 and rows['new_shape']['counters'].get('fxgraph_cache_hit')==1
    assert rows['new_shape']['cache_key']==rows['cold']['cache_key']
    assert rows['new_dtype']['calls']['lowering']==1 and rows['new_dtype']['counters'].get('fxgraph_cache_miss')==1
    assert rows['new_dtype']['cache_key']!=rows['cold']['cache_key']
    assert rows['bypass']['calls']['lowering']==1 and rows['bypass']['counters'].get('fxgraph_cache_bypass')==1
    assert all(r['calls']['post_compile']==1 for r in rows.values())
    report={'cache_directory':cache,'workers':rows}
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit
label=sys.argv[2];cache=sys.argv[3]
os.environ['TORCHINDUCTOR_CACHE_DIR']=cache
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.codecache import FxGraphCache
import importlib
cache_module=importlib.import_module('torch._inductor.codecache')
from torch._dynamo.utils import counters
from torch._inductor.virtualized import V
from contextlib import nullcontext
shape=(3,4) if label=='new_shape' else (2,3)
dtype=torch.float64 if label=='new_dtype' else torch.float32
x=torch.arange(shape[0]*shape[1],dtype=dtype).reshape(shape)+(100 if label=='warm' else 0)
y=torch.ones_like(x)
fn=lambda x,y:(torch.relu(x+y)*2,)
gm=make_fx(fn)(x,y)
counters['inductor'].clear();calls={'lowering':0,'post_compile':0}
orig_run=GraphLowering.run;orig_post=FxGraphCache.post_compile;orig_hash=cache_module.compiled_fx_graph_hash
def graph_hash(*args,**kwargs):
    result=orig_hash(*args,**kwargs)
    (OUT/f'{label}_hash.txt').write_text('\n'.join(result[1])+'\n')
    return result
cache_module.compiled_fx_graph_hash=graph_hash

def run(self,*args):calls['lowering']+=1;return orig_run(self,*args)
def post(*args,**kwargs):calls['post_compile']+=1;return orig_post(*args,**kwargs)
GraphLowering.run=run;FxGraphCache.post_compile=staticmethod(post)
mode=FakeTensorMode(shape_env=ShapeEnv(),allow_non_fake_inputs=True)
ctx=nullcontext() if label=='bypass' else torch._guards.tracing(torch._guards.TracingContext(mode))
try:
    with config.patch({'force_disable_caches':False,'fx_graph_cache':True,'fx_graph_remote_cache':False,'triton.cudagraphs':False}),ctx:
        compiled=compile_fx_inner(gm,[mode.from_tensor(x),mode.from_tensor(y)],is_inference=True)
finally:GraphLowering.run=orig_run;FxGraphCache.post_compile=staticmethod(orig_post);cache_module.compiled_fx_graph_hash=orig_hash
actual=compiled([x,y]);torch.testing.assert_close(actual,fn(x,y))
row={'example_kind':'FakeTensor','pid':os.getpid(),'calls':calls,'counters':{k:v for k,v in counters['inductor'].items() if k.startswith('fxgraph_cache_')},'input':x.tolist(),'output':actual[0].tolist(),'shape':list(shape),'dtype':str(dtype),'cache_key':compiled._fx_graph_cache_key,'callable_loaded':callable(compiled.current_callable),'check':'numeric pass'}
(OUT/f'{label}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
(OUT/f'{label}_output.py').write_text(compiled.source_code)
print(json.dumps(row,ensure_ascii=False,indent=2))
