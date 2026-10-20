"""融合大小、缓冲区复用和生成驱动的受控对照；不是性能计时。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/fusion-lifetime';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-fusion-lifetime')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config,metrics
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V
x=torch.arange(6,dtype=torch.float32).reshape(2,3);y=torch.ones_like(x)
mat=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
cases={
'fusion_allowed':(lambda x,y:(x+y,(x+y)*2),[x,y],{'max_fusion_size':64}),
'fusion_limited':(lambda x,y:(x+y,(x+y)*2),[x,y],{'max_fusion_size':1}),
'reuse_allowed':(lambda x,y:(torch.relu(x@y),),[x,mat],{'inplace_buffers':True,'allow_buffer_reuse':True}),
'reuse_disabled':(lambda x,y:(torch.relu(x@y),),[x,mat],{'inplace_buffers':False,'allow_buffer_reuse':False})}
orig_fuse=Scheduler.fuse_nodes;orig_init=Scheduler._init;orig_codegen=Scheduler._codegen;orig_save=GraphLowering.save_output_code

def fuse(self,nodes):
    before=[{'name':n.get_name(),'kind':type(n).__name__} for n in nodes]
    r=orig_fuse(self,nodes)
    state['fusion']={'before':before,'after':[{'name':n.get_name(),'kind':type(n).__name__} for n in r]}
    return r

def init(self,nodes):
    r=orig_init(self,nodes)
    state['last_usage']={n.get_name():list(n.last_usage) for n in self.nodes}
    return r

def codegen(self):
    r=orig_codegen(self)
    state['after_codegen']={'completed_operations':list(self.completed_operations),'available_buffers':list(self.available_buffer_names),'wrapper_allocated':sorted(V.graph.wrapper_code.allocated),'wrapper_freed':sorted(V.graph.wrapper_code.freed),'removed_buffers':sorted(V.graph.removed_buffers)}
    return r
Scheduler.fuse_nodes=fuse;Scheduler._init=init;Scheduler._codegen=codegen
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/f'{case}_output.py').write_text(code))
report={}
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False}):
        for case,(fn,inputs,patch) in cases.items():
            state={};gm=make_fx(fn)(*inputs);metrics.reset()
            with config.patch(patch):compiled=compile_fx_inner(gm,inputs,is_inference=True)
            actual_inputs=[t.clone() for t in inputs]
            before=[t.clone() for t in actual_inputs]
            expected=fn(*[t.clone() for t in inputs])
            actual=compiled(list(actual_inputs))
            torch.testing.assert_close(actual,expected)
            assert all(torch.equal(t,b) for t,b in zip(actual_inputs,before))
            assert all(o.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for o in actual for t in actual_inputs)
            report[case]={'config':patch,**state,'generated_kernel_count':metrics.generated_kernel_count,'outputs':[t.tolist() for t in actual],'check':'numeric, input preservation and aliases pass'}
finally:
    Scheduler.fuse_nodes=orig_fuse;Scheduler._init=orig_init;Scheduler._codegen=orig_codegen;GraphLowering.save_output_code=staticmethod(orig_save)
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
