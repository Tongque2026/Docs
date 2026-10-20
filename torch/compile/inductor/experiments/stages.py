"""采集真实 post-grad、Lowering 与调度交接；所有修改只在进程内。"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/stages'
OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-stages')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import importlib,json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config,metrics
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
cfx=importlib.import_module('torch._inductor.compile_fx')

def dump(name,text): (OUT/f'{case}_{name}.txt').write_text(text+'\n')
def graph_dump(gm):
    rows=[str(gm.graph),'']
    for n in gm.graph.nodes:
        v=n.meta.get('val')
        if isinstance(v,torch.Tensor): rows.append(f'{n.name}: {type(v).__name__} shape={tuple(v.shape)} stride={v.stride()} dtype={v.dtype} device={v.device}')
    return '\n'.join(rows)
orig_post=cfx.post_grad_passes
orig_run=GraphLowering.run
orig_fuse=Scheduler.fuse_nodes
orig_init=Scheduler._init
orig_save=GraphLowering.save_output_code

def post(gm,is_inference):
    dump('before_post',graph_dump(gm))
    result=orig_post(gm,is_inference)
    dump('after_post',graph_dump(gm))
    return result

def run(self,*args):
    result=orig_run(self,*args)
    dump('lowered','\n'.join(repr(o) for o in self.operations))
    return result

def fuse(self,nodes):
    dump('before_fusion','\n\n'.join(n.debug_str() for n in nodes))
    result=orig_fuse(self,nodes)
    dump('after_fusion','\n\n'.join(n.debug_str() for n in result))
    return result

def init(self,nodes):
    result=orig_init(self,nodes)
    dump('last_usage','\n'.join(f'{n.get_name()}: {list(n.last_usage)}' for n in self.nodes))
    return result

cfx.post_grad_passes=post
GraphLowering.run=run
Scheduler.fuse_nodes=fuse
Scheduler._init=init
GraphLowering.save_output_code=staticmethod(lambda code: (OUT/f'{case}_output.py').write_text(code))
x=torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
y=torch.ones_like(x)
cases={
    'clone_internal':(lambda x,y:(torch.relu(x.clone())+y,),[x,y]),
    'clone_output':(lambda x,y:(x.clone(),),[x,y]),
    'view':(lambda x,y:(x.view(6),),[x,y]),
    'reduction':(lambda x,y:((x+y).sum(1),),[x,y]),
    'reduction_large':(lambda x,y:((x+y).sum(1),),[torch.arange(128,dtype=torch.float32).reshape(2,64),torch.ones(2,64)]),
    'mm':(lambda x,y:(torch.relu(x@y),),[x,torch.tensor([[1.,2.],[3.,4.],[5.,6.]])]),
    'multiple_outputs':(lambda x,y:(x+y,(x+y)*2),[x,y]),
}
results={}
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):
        for case,(fn,inputs) in cases.items():
            gm=make_fx(fn)(*inputs)
            dump('input',graph_dump(gm))
            before=[i.clone() for i in inputs]
            metrics.reset()
            compiled=cfx.compile_fx_inner(gm,inputs,is_inference=True)
            actual=compiled(list(inputs))
            expected=fn(*inputs)
            torch.testing.assert_close(actual,expected)
            assert all(torch.equal(a,b) for a,b in zip(inputs,before))
            results[case]={'generated_kernels':metrics.generated_kernel_count,'output':[a.tolist() for a in actual],'strides':[list(a.stride()) for a in actual], 'aliases_input0':[a.untyped_storage().data_ptr()==inputs[0].untyped_storage().data_ptr() for a in actual]}
finally:
    cfx.post_grad_passes=orig_post
    GraphLowering.run=orig_run
    Scheduler.fuse_nodes=orig_fuse
    Scheduler._init=orig_init
    GraphLowering.save_output_code=staticmethod(orig_save)
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
