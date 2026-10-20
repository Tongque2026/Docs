"""编译并运行索引、物化与写入案例，采集真实 Lowering 和依赖。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/expressions-storage';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-expressions-storage')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config,ir,metrics
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V

def copy_old(x,y):
    old=x+1
    x.copy_(y)
    return (old,x)
def mutate_only(x,y):
    x.copy_(y)
    return (y+1,)
def dead_old(x,y):
    unused=x+1
    x.copy_(y)
    return (x,)
def heavy(x):
    a=x.exp()
    return (a+1,a*2)
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
cases={
'broadcast':(lambda x,y:(x+y,),[x,torch.tensor([[10.,20.,30.]])]),
'permute':(lambda x,y:(x.permute(1,0)+y,),[x,torch.tensor([10.,20.])]),
'expand':(lambda x:(x.expand(2,3),),[torch.tensor([[10.,20.,30.]])]),
'reshape_copy':(lambda x:(x.permute(1,0).reshape(6),),[x]),
'reuse_exp':(heavy,[x]),
'copy_old':(copy_old,[torch.arange(4,dtype=torch.float32),torch.full((4,),9.)]),
'mutate_only':(mutate_only,[torch.arange(4,dtype=torch.float32),torch.full((4,),9.)]),
'dead_old':(dead_old,[torch.arange(4,dtype=torch.float32),torch.full((4,),9.)]),
}
clone=lambda a:[t.clone() for t in a]
def aliases(outputs,inputs):return [[o.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in inputs] for o in outputs]
def dump(s,t): (OUT/f'{case}_{s}.txt').write_text(t+'\n')
orig_run=GraphLowering.run;orig_node=GraphLowering.run_node;orig_dep=Scheduler.compute_dependencies;orig_fuse=Scheduler.fuse_nodes;orig_save=GraphLowering.save_output_code

def run(self,*args):
    result=orig_run(self,*args)
    dump('lowered','\n'.join(repr(o) for o in self.operations))
    return result

def node(self,n):
    r=orig_node(self,n);v=r;chain=[]
    while True:
        chain.append(type(v).__name__)
        if isinstance(v,(ir.TensorBox,ir.StorageBox)):v=v.data
        else:break
    events.append({'name':n.name,'target':str(n.target),'chain':chain,'operations':len(self.operations)})
    return r

def dep(self):
    r=orig_dep(self)
    dump('dependencies','\n\n'.join(n.debug_str() for n in self.nodes))
    depstate.update({'mutation_renames':dict(self.mutation_renames),'mutation_real_name':dict(self.mutation_real_name),'mutated_inputs':list(V.graph.mutated_inputs)})
    return r

def fuse(self,nodes):
    dump('before_fusion','\n\n'.join(n.debug_str() for n in nodes))
    r=orig_fuse(self,nodes)
    dump('after_fusion','\n\n'.join(n.debug_str() for n in r))
    return r

GraphLowering.run=run;GraphLowering.run_node=node;Scheduler.compute_dependencies=dep;Scheduler.fuse_nodes=fuse
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/f'{case}_output.py').write_text(code))
report={}
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):
        for case,(fn,inputs) in cases.items():
            events=[];depstate={}
            gm=make_fx(fn)(*clone(inputs));dump('input',str(gm.graph))
            metrics.reset();compiled=compile_fx_inner(gm,clone(inputs),is_inference=True)
            ai=clone(inputs);ei=clone(inputs)
            actual=compiled(list(ai));expected=fn(*ei)
            torch.testing.assert_close(actual,expected)
            for a,e in zip(ai,ei):torch.testing.assert_close(a,e)
            assert aliases(actual,ai)==aliases(expected,ei)
            report[case]={'outputs':[o.tolist() for o in actual],'shapes':[list(o.shape) for o in actual],'strides':[list(o.stride()) for o in actual],'inputs_after':[i.tolist() for i in ai],'output_input_aliases':aliases(actual,ai),'generated_kernels':metrics.generated_kernel_count,'events':events,'dependencies_state':depstate,'mutated_input_idxs':sorted(compiled.mutated_input_idxs),'check':'numeric, input mutation and aliases pass'}
finally:
    GraphLowering.run=orig_run;GraphLowering.run_node=orig_node;Scheduler.compute_dependencies=orig_dep;Scheduler.fuse_nodes=orig_fuse;GraphLowering.save_output_code=staticmethod(orig_save)
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{x:v[x] for x in ['generated_kernels','strides','mutated_input_idxs','check']} for k,v in report.items()},ensure_ascii=False,indent=2))
