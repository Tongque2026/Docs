"""真实 selector 的分支和延迟协议。lookup 数值为明确的排序槽位 fixture，绝非设备耗时。"""
from pathlib import Path
import json,types
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config,ir
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._inductor.select_algorithm import AlgorithmSelectorCache,NoValidChoicesError
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
R=Path(__file__).resolve().parents[1];O=R/'experiments/algorithm-selection';O.mkdir(exist_ok=True)
class Caller:
 def __init__(self,name):self.name=name;self.output_calls=0
 def hash_key(self):return self.name
 def output_node(self):self.output_calls+=1;return self.name
class CudaCaller(CUDATemplateCaller):
 def __init__(self,name):self.name=name;self.output_calls=0
 def hash_key(self):return self.name
 def output_node(self):self.output_calls+=1;return self.name

def selector(mapping):
 s=object.__new__(AlgorithmSelectorCache);s.precompile_cache={};s.feedback_saver_fns=[];events=[]
 def lookup(choices,name,key,benchmark):events.append({'benchmark_callback_provided':benchmark is not None});return {c:mapping[c.name] for c in choices if c.name in mapping}
 s.lookup=lookup
 s.make_benchmark_fn=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('protocol must never benchmark'))
 return s,events
x=torch.ones(2,2);gm=make_fx(lambda x:(x,))(x);g=GraphLowering(gm,[x],layout_opt=False)
reports={}
with V.set_graph_handler(g),config.patch({'max_autotune_gemm':True,'autotune_in_subproc':False,'trace.log_autotuning_results':False}):
 layout_cpu=ir.FixedLayout(torch.device('cpu'),torch.float32,[2,2]);layout_cuda=ir.FixedLayout(torch.device('cuda',0),torch.float32,[2,2]);inp=ir.InputBuffer('input',layout_cuda)
 def call(s,choices,layout=layout_cpu,**kwargs):return s('protocol',choices,[inp],layout,precompilation_timeout_seconds=0,**kwargs)
 a,b=Caller('a'),Caller('b');s,events=selector({})
 assert call(s,[None,a])=='a' and not events
 reports['filtered_single']={'output':'a','lookup_calls':len(events)}
 try:call(s,[None])
 except NoValidChoicesError:reports['empty_filtered']={'error':'NoValidChoicesError'}
 else:raise AssertionError('empty choices accepted')
 for name,mapping,expected in [('empty_timing_map',{},'a'),('partial_second_only',{'b':1.},'a'),('full_rank_map',{'a':2.,'b':1.},'b')]:
  s,events=selector(mapping);actual=call(s,[a,b]);assert actual==expected
  reports[name]={'output':actual,'lookup_calls':len(events),'ranking_slots':mapping}
 s,events=selector({'a':float('inf'),'b':float('inf')})
 try:call(s,[a,b])
 except NoValidChoicesError:reports['all_failed']={'error':'NoValidChoicesError','lookup_calls':len(events)}
 else:raise AssertionError('all failed candidates accepted')
 # Single CUDATemplateCaller does not use normal single-candidate fast path.
 cuda=CudaCaller('workspace_choice');s,events=selector({'workspace_choice':1.})
 assert call(s,[cuda])=='workspace_choice' and len(events)==1
 reports['single_cuda_exception']={'lookup_calls':len(events),'output_calls':cuda.output_calls,'scope':'actual isinstance exception with metadata-only subclass, no CUDA caller compilation or workspace allocation'}
 # Real MultiTemplateBuffer, no benchmark or GPU data. Ranking map is a fixture.
 s,events=selector({'a':2.,'b':1.});result=call(s,[a,b],layout_cuda,return_multi_template=True);multi=result.data.data
 assert isinstance(multi,ir.MultiTemplateBuffer) and len(events)==0
 first=multi.choice_timings;second=multi.choice_timings;assert first is second and len(events)==1
 minimum=multi.get_min_choice();assert minimum==(b,1.) and len(events)==1
 reports['deferred_once']={'kind':type(multi).__name__,'lookup_before_property':0,'lookup_after_two_reads':1,'min_rank_choice':minimum[0].name,'scope':'actual deferred buffer with synthetic ordering slots; no timing or GPU kernel execution'}
 for name,layout,extra in [('cpu_disables_defer',layout_cpu,{}),('input_generator_disables_defer',layout_cuda,{'input_gen_fns':{}})]:
  s,events=selector({'a':2.,'b':1.});actual=call(s,[a,b],layout,return_multi_template=True,**extra)
  assert actual=='b' and len(events)==1
  reports[name]={'output':actual,'lookup_calls':len(events)}
reports['scope']='real selector control flow with Caller/output/lookup fixtures; finite values 1 and 2 are synthetic ranking slots, not milliseconds or device performance; no benchmark function allowed'
(O/'protocol.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n');print(json.dumps(reports,ensure_ascii=False,indent=2))
