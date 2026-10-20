"""真实 wrapper 生成 API 的分支；设备正文使用显式 stub 协议，CPU 完整编译另列。"""
from pathlib import Path
import os,json,types,contextlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/wrapper-devices';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'experiments/cache-wrapper-devices');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch,sympy
from torch._inductor import config,ir
from torch._inductor.codegen.wrapper import WrapperCodeGen,WrapperLine,EnterDeviceContextManagerLine,ExitDeviceContextManagerLine
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.virtualized import V
from torch._inductor.utils import IndentedBuffer
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch.fx.experimental.proxy_tensor import make_fx

class Ops(DeviceOpOverrides):
    def import_get_raw_stream_as(self,name):return f'from lesson_runtime import get_raw_stream as {name}'
    def device_guard(self,index):return f'device_api.guard({index})'
    def set_device(self,index):return f'device_api.set_device({index})'
    def synchronize(self):return 'device_api.synchronize()'
class Sizes:
    def simplify(self,x):return sympy.simplify(x)
    def size_hints(self,x,**kwargs):return tuple(int(v) for v in x)
    def size_hint(self,x,**kwargs):return int(x)
    inv_precomputed_replacements={}
class Fixture:
    def __init__(self):
        self.aot_mode=False;self.cpp_wrapper=False;self.constant_reprs={};self.graph_inputs={};self.graph_input_names=[];self.device_ops=Ops();self.sizevars=Sizes();self.mutated_buffers=set();self.buffers={};self.scheduler=types.SimpleNamespace(get_current_device_or_throw=lambda:torch.device('cuda',0))
    def is_unspec_arg(self,name):return name=='scalar'
    def try_get_buffer(self,name):return self.buffers.get(name)
    def get_buffer(self,name):return self.buffers[name]
def make(graph):
    with V.set_graph_handler(graph):return WrapperCodeGen()
def render(wrapper,indent=0):
    out=IndentedBuffer(initial_indent=indent)
    for line in wrapper.lines:
        if isinstance(line,WrapperLine):line.codegen(out)
        else:out.writeline(line)
    return out.getvalue()
report={}
# 同一图/设备只生成一条 stream 赋值，不缓存运行时返回的 stream 值。
g=Fixture();w=make(g)
with config.patch({'triton.autotune_at_compile_time':False}),V.set_graph_handler(g):
    w.codegen_device_guard_enter(0)
    w.generate_kernel_call('kernel_a',['x','scalar'],grid=[6],device_index=None)
    w.generate_kernel_call('kernel_b',['x','scalar'],grid=[6],device_index=0)
    w.codegen_device_guard_exit()
    body=render(w,1)
(OUT/'protocol_body.py').write_text('def generated(x, scalar):\n'+body)
assert body.count('stream0 = get_raw_stream(0)')==1 and 'scalar.item()' in body
runtime_events=[]
class Runtime:
    @contextlib.contextmanager
    def guard(self,index):
        runtime_events.append(['guard_enter',index]);yield;runtime_events.append(['guard_exit',index])
    def set_device(self,index):runtime_events.append(['set_device',index])
class Kernel:
    def __init__(self,name):self.name=name
    def run(self,*args,grid,stream):runtime_events.append([self.name,list(grid),stream,args[1]])
def stream(index):runtime_events.append(['stream',index]);return 123+index
namespace={'device_api':Runtime(),'get_raw_stream':stream,'kernel_a':Kernel('a'),'kernel_b':Kernel('b'),'grid':lambda *dims:dims}
exec((OUT/'protocol_body.py').read_text(),namespace)
namespace['generated'](torch.ones(6),torch.tensor(4));first=list(runtime_events);namespace['generated'](torch.ones(6),torch.tensor(5))
assert [e[0] for e in first]==['guard_enter','set_device','stream','a','b','guard_exit']
assert sum(e[0]=='stream' for e in runtime_events)==2
report['stream_protocol']={'body':body,'first_events':first,'stream_queries_two_invocations':2,'scope':'actual generated body executed with explicit Python stub guard/stream/kernel; not GPU execution'}
# graph 是代码生成缓存 key 的一部分；相同 wrapper 不同 graph 会再生成赋值。
g2=Fixture();w=make(g)
with V.set_graph_handler(g):
    a=w.write_get_raw_stream(0,g);b=w.write_get_raw_stream(0,g);c=w.write_get_raw_stream(0,g2)
report['stream_cache_keys']={'names':[a,b,c],'assignments':sum(isinstance(v,str) and 'get_raw_stream(0)' in v for v in w.lines),'import_count':w.imports.getvalue().count('from lesson_runtime import get_raw_stream')}
assert report['stream_cache_keys']['assignments']==2 and report['stream_cache_keys']['import_count']==1
for name,cuda,triton in [('cpu_call',False,True),('cuda_cpp_call',True,False),('triton_callable_grid',True,True)]:
    w=make(g)
    with V.set_graph_handler(g):
        w.generate_kernel_call('compiled_kernel',['x','out'],cuda=cuda,triton=triton,device_index=2,grid=None if name=='triton_callable_grid' else [8])
        text=render(w)
    report[name]={'body':text,'scope':'generated source only; no kernel executed'};(OUT/f'{name}.txt').write_text(text)
assert report['cpu_call']['body']=='compiled_kernel(x, out)\n'
assert 'compiled_kernel.compiled_kernel(x, out, c_void_p(stream2))' in report['cuda_cpp_call']['body']
assert 'grid=grid, stream=stream2' in report['triton_callable_grid']['body']
# 编译期 autotune 另有调用区，同 kernel 的样例启动只生成一次。
g=Fixture();layout=ir.FixedLayout(torch.device('cuda',0),torch.float32,[6],[1]);g.buffers={'x':ir.InputBuffer('x',layout),'out':ir.InputBuffer('out',layout)}
with config.patch({'triton.autotune_at_compile_time':True}),V.set_graph_handler(g):
    w=WrapperCodeGen();w.codegen_device_guard_enter(0)
    for _ in range(2):w.generate_kernel_call('tuned',['x','out'],grid=[6],arg_types=[torch.float32,torch.float32])
    w.codegen_device_guard_exit();main=render(w);autotune=w.kernel_autotune_calls.getvalue()
report['autotune_sections']={'main_launches':main.count('tuned.run('),'example_launches':autotune.count('tuned.run('),'main':main,'autotune':autotune,'scope':'source generation only; sample tensors and autotune not executed'}
assert (report['autotune_sections']['main_launches'],report['autotune_sections']['example_launches'])==(2,1)
(OUT/'autotune_source.txt').write_text(autotune)
with config.patch({'triton.autotune_at_compile_time':True}),V.set_graph_handler(g):
    w=WrapperCodeGen()
    try:w.generate_kernel_call('bad',['x'],grid=[6],arg_types=[])
    except AssertionError as e:report['argument_types_rejected']={'error':type(e).__name__,'message':str(e)}
    else:raise AssertionError('expected arg types mismatch')
# C++/AOT guard 分支使用真实 line.codegen；只检查输出字符串及单设备断言。
for name,aot,abi,previous,index in [('cpp_jit_first',False,True,None,2),('cpp_jit_next',False,True,2,3),('cpp_jit_native',False,False,None,2),('aot_first',True,True,None,2),('aot_native',True,False,None,2),('aot_same',True,True,2,2),('aot_mismatch',True,True,2,3)]:
    gg=Fixture();gg.cpp_wrapper=True;gg.aot_mode=aot;buf=IndentedBuffer()
    with config.patch({'abi_compatible':abi}),V.set_graph_handler(gg):
        try:EnterDeviceContextManagerLine(index,previous).codegen(buf)
        except AssertionError as e:report[name]={'error':type(e).__name__,'message':str(e)};assert name=='aot_mismatch'
        else:report[name]={'source':buf.getvalue(),'scope':'C++ source emission only; no native compilation or execution'}
assert 'AOTICudaStreamGuard' in report['aot_first']['source'] and 'CUDAStreamGuard' in report['aot_native']['source']
assert 'set_index(3)' in report['cpp_jit_next']['source'] and report['aot_same']['source'].strip()==''
# Prefix 的符号绑定、size checks、debug 同步都来自真实 generator。
n=sympy.Symbol('n',integer=True,positive=True);gg=Fixture();gg.graph_inputs={'x':ir.InputBuffer('x',ir.FixedLayout(torch.device('cpu'),torch.float32,[n,3],[3,1]))};gg.graph_input_names=['x']
with config.patch({'triton.debug_sync_graph':True,'size_asserts':True}),V.set_graph_handler(gg):w=WrapperCodeGen()
report['prefix']={'source':w.prefix.getvalue(),'scope':'source generation only'}
assert 'x, = args' in report['prefix']['source'] and 'args.clear()' in report['prefix']['source'] and 'x_size = x.size()' in report['prefix']['source'] and 'n = x_size[0]' in report['prefix']['source']
assert 'device_api.synchronize()' in report['prefix']['source']
# 完整 CPU 编译：list 被清空，但调用者另外持有的 Tensor 数据保持不变。
x=torch.arange(6,dtype=torch.float32).reshape(2,3);fn=lambda x:(x+1,);gm=make_fx(fn)(x);original=GraphLowering.save_output_code
GraphLowering.save_output_code=staticmethod(lambda code:(OUT/'cpu_output.py').write_text(code))
try:
    with config.patch({'force_disable_caches':True,'fx_graph_cache':False,'triton.cudagraphs':False}):compiled=compile_fx_inner(gm,[x.clone()],is_inference=True)
finally:GraphLowering.save_output_code=staticmethod(original)
input_tensor=x.clone();args=[input_tensor];actual=compiled(args);torch.testing.assert_close(actual,fn(x));torch.testing.assert_close(input_tensor,x);assert args==[]
failures={}
for name,bad in [('wrong_shape',torch.zeros(3,2)),('wrong_stride',torch.arange(6,dtype=torch.float32).reshape(3,2).t())]:
    try:compiled([bad])
    except AssertionError as e:failures[name]={'error':type(e).__name__,'message':str(e)}
    else:raise AssertionError(name)
report['cpu_compile']={'output':actual[0].tolist(),'args_cleared':args==[],'input_tensor_unchanged':True,'layout_rejections':failures,'scope':'complete CPU compilation and execution'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
