"""真实 CPU ATEN/CPP 双候选计时与跨进程计时缓存；不使用人工时间。"""
from pathlib import Path
import os,sys,json,tempfile,subprocess
R=Path(__file__).resolve().parents[1];O=R/'experiments/algorithm-selection';O.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
 cache_parent=R/'experiments/cache-algorithm-selection';cache_parent.mkdir(exist_ok=True)
 private_cache=tempfile.mkdtemp(prefix='real-cpu-',dir=cache_parent)
 reports={}
 for phase in ['cold','warm']:
  env=dict(os.environ,PYTHONPATH='/work/pytorch',TORCHINDUCTOR_CACHE_DIR=private_cache,TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1')
  proc=subprocess.run([sys.executable,__file__,'--worker',phase],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
  (O/f'{phase}.stdout.txt').write_text(proc.stdout);(O/f'{phase}.stderr.txt').write_text(proc.stderr)
  assert proc.returncode==0,(phase,proc.stderr[-3000:])
  reports[phase]=json.loads((O/f'{phase}.json').read_text())
 assert reports['cold']['benchmark_fn_calls']==1 and reports['warm']['benchmark_fn_calls']==0
 assert reports['cold']['timings_ms']==reports['warm']['timings_ms']
 assert reports['cold']['candidate_hashes']==reports['warm']['candidate_hashes']
 assert reports['cold']['output_sample']!=reports['warm']['output_sample']
 reports['scope']='real native CPU candidates and unmodified selector benchmark; same private timing cache shared by two fresh processes; warm model input values differ'
 (O/'real-cpu.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n');print(json.dumps({p:{k:v for k,v in x.items() if k in ['benchmark_fn_calls','timings_ms','selected_ir','output_sample','precompile_calls']} for p,x in reports.items() if isinstance(x,dict)},ensure_ascii=False,indent=2));sys.exit(0)
phase=sys.argv[-1]
import torch,math
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.codegen.cpp_template_kernel import CppTemplateCaller
from torch._inductor.codecache import CacheBase
import torch._inductor.select_algorithm as sa
torch.set_num_threads(1)
W=torch.arange(64*64,dtype=torch.float32).reshape(64,64)/4096;weight_before=W.clone()
x=torch.arange(16*64,dtype=torch.float32).reshape(16,64)/1024
if phase=='warm':x=x+1.5
def fn(x):return (x@W,)
expected=fn(x);state={'phase':phase,'benchmark_fn_calls':0,'precompile_calls':[],'timings_ms':{},'candidate_hashes':{}}
orig_select=sa.AlgorithmSelectorCache.__call__;orig_bench=sa.AlgorithmSelectorCache.make_benchmark_fn;orig_pre=CppTemplateCaller.precompile;orig_save=GraphLowering.save_output_code;orig_verify=sa.VERIFY

def select(self,name,choices,input_nodes,layout,*args,**kwargs):
 state['candidate_hashes']={c.name:c.hash_key() for c in choices}
 state['candidate_kinds']={c.name:type(getattr(c,'_wrapped',c)).__name__ for c in choices}
 assert len(choices)==2
 def feedback(timings,*args):state['timings_ms']={c.name:float(ms) for c,ms in timings.items()}
 self.feedback_saver_fns.append(feedback)
 result=orig_select(self,name,choices,input_nodes,layout,*args,**kwargs)
 state['selected_ir']=type(result.data.data).__name__
 return result

def make_benchmark(cls,*args,**kwargs):state['benchmark_fn_calls']+=1;return orig_bench(*args,**kwargs)
def precompile(self):state['precompile_calls'].append(self.name);return orig_pre(self)
sa.AlgorithmSelectorCache.__call__=select;sa.AlgorithmSelectorCache.make_benchmark_fn=classmethod(make_benchmark);CppTemplateCaller.precompile=precompile
GraphLowering.save_output_code=staticmethod(lambda code:(O/f'{phase}_output.py').write_text(code));sa.VERIFY={'atol':1e-3,'rtol':1e-3}
try:
 gm=make_fx(fn)(x)
 with config.patch({'force_disable_caches':False,'fx_graph_cache':False,'triton.cudagraphs':False,'benchmark_fusion':False,'max_autotune':False,'max_autotune_gemm':True,'max_autotune_gemm_backends':'ATEN,CPP','autotune_fallback_to_aten':False,'cpp.threads':1,'autotune_local_cache':True,'global_cache_dir':None,'autotune_in_subproc':False}):
  compiled=compile_fx_inner(gm,[x],is_inference=True)
 actual_input=x.clone();actual=compiled([actual_input]);torch.testing.assert_close(actual,expected)
 assert torch.equal(actual_input,x) and torch.equal(W,weight_before)
 assert actual[0].untyped_storage().data_ptr()!=actual_input.untyped_storage().data_ptr()
 assert len(state['timings_ms'])==2 and all(math.isfinite(v) and v>0 for v in state['timings_ms'].values())
 selected=min(state['timings_ms'],key=state['timings_ms'].get);state['selected_choice']=selected
 expected_ir='ExternKernelOut' if state['candidate_kinds'][selected]=='ExternKernelCaller' else 'CppTemplateBuffer'
 assert state['selected_ir']==expected_ir
 state.update(verify_enabled=True,output_sample=actual[0][0,:4].tolist(),output_shape=list(actual[0].shape),input_preserved=True,output_aliases_input=False,captured_weight_preserved=True,cache_path=str(CacheBase.get_local_cache_path()),system=CacheBase.get_system(),python_version=sys.version,scope='real native CPU benchmark and execution; VERIFY applies when candidates actually benchmark; cache hit skips benchmark/VERIFY')
 (O/f'{phase}.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');print(json.dumps(state,ensure_ascii=False,indent=2))
finally:
 sa.AlgorithmSelectorCache.__call__=orig_select;sa.AlgorithmSelectorCache.make_benchmark_fn=orig_bench;CppTemplateCaller.precompile=orig_pre;GraphLowering.save_output_code=staticmethod(orig_save);sa.VERIFY=orig_verify
