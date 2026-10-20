"""自定义 fallback、条件和循环的单图编译实验；全部运行在 CPU。"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/special-ops';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-special-ops')
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._higher_order_ops.cond import cond_op
from torch._higher_order_ops.while_loop import while_loop_op

lib=torch.library.Library('inductor_learning_examples','DEF')
lib.define('pair(Tensor x) -> (Tensor, Tensor)')
lib.impl('pair',lambda x:(x+1,x*2),'CPU')
@torch.library.register_fake('inductor_learning_examples::pair')
def fake_pair(x):return torch.empty_like(x),torch.empty_like(x)
def fallback_case(x):
    a,b=torch.ops.inductor_learning_examples.pair.default(x)
    return (torch.relu(a),b)
def conditional(p,x):return cond_op(p,lambda x:(x+1,),lambda x:(x*2,),(x,))
def loop(i,x):return while_loop_op(lambda i,x:i<3,lambda i,x:(i+1,x+1),(i,x),())
x=torch.tensor([-2.,0.,3.])
report={};original_run=GraphLowering.run;original_save=GraphLowering.save_output_code

def run(self,*args):
    result=original_run(self,*args)
    scope=self.name or 'root'
    (OUT/f'{case}_{scope}_lowered.txt').write_text('\n'.join(repr(o) for o in self.operations)+'\n')
    lowerings.append({'scope':scope,'operations':[type(o).__name__ for o in self.operations],'never_reuse_buffers':sorted(self.never_reuse_buffers)})
    return result
GraphLowering.run=run
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/f'{case}_output.py').write_text(code))
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):
        case='fallback_forbidden';lowerings=[]
        gm=make_fx(fallback_case)(x)
        try:
            with config.patch({'implicit_fallbacks':False}):compile_fx_inner(gm,[x],is_inference=True)
        except Exception as e:report[case]={'error':type(e).__name__,'message':str(e)[:250]}
        else:raise AssertionError('missing custom lowering must fail when fallback is disabled')
        cases={
            'fallback':(fallback_case,[x],[[x]]),
            'cond':(conditional,[torch.tensor(True),x],[[torch.tensor(True),x],[torch.tensor(False),x]]),
            'while':(loop,[torch.tensor(0),x],[[torch.tensor(0),x],[torch.tensor(3),x]])}
        for case,(fn,example,executions) in cases.items():
            lowerings=[];gm=make_fx(fn)(*example)
            (OUT/f'{case}_input.txt').write_text(str(gm.graph)+'\n')
            with config.patch({'implicit_fallbacks':True}):compiled=compile_fx_inner(gm,example,is_inference=True)
            rows=[]
            for inputs in executions:
                ai=[i.clone() for i in inputs];ei=[i.clone() for i in inputs]
                actual=compiled(list(ai));expected=fn(*ei)
                torch.testing.assert_close(actual,expected)
                for a,e in zip(ai,ei):torch.testing.assert_close(a,e)
                alias=lambda o,ins:[o.untyped_storage().data_ptr()==i.untyped_storage().data_ptr() for i in ins]
                assert [alias(o,ai) for o in actual]==[alias(o,ei) for o in expected]
                rows.append({'inputs':[i.tolist() for i in inputs],'outputs':[o.tolist() for o in actual],'output_input_aliases':[alias(o,ai) for o in actual],'check':'numeric, inputs and aliases pass'})
            report[case]={'lowerings':lowerings,'runs':rows}
finally:
    GraphLowering.run=original_run;GraphLowering.save_output_code=staticmethod(original_save)
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
