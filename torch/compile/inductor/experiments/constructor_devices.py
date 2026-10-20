"""真实 ConstructorMoverPass 的设备/依赖分支；所有 CUDA 值仅为 FakeTensor。"""
from pathlib import Path
import json
import torch
from torch._subclasses.fake_tensor import FakeTensor,FakeTensorMode
from torch._inductor.fx_passes.post_grad import ConstructorMoverPass,is_index_put_and_requires_h2d_sync_for_gpu_value
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/constructor-devices';OUT.mkdir(exist_ok=True)
a=torch.ops.aten;mode=FakeTensorMode()
def val(shape,device,dtype=torch.float32):return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype),torch.device(device))
def opaque(x):return x+1
report={}
for case in ['movable','output_blocked','output_allowed','no_target','multiple_targets','no_device_kwarg','string_device','cpu_input_blocked','two_constructors','linked_output_blocked','repeated_input','opaque_blocked','gpu_boundary_two_inputs']:
    g=torch.fx.Graph();p=g.placeholder('p');p.meta['val']=val([2],'cpu' if case=='no_target' else 'cuda:0')
    if case=='multiple_targets':q=g.placeholder('q');q.meta['val']=val([2],'cuda:1')
    def constructor(name,device_arg=torch.device('cpu'),dtype=torch.float32):
        kwargs={'dtype':dtype}
        if device_arg is not None:kwargs['device']=device_arg
        c=g.call_function(a.full.default,([2],1),kwargs);c.name=name;c.meta['val']=val([2],'cpu',dtype);return c
    c=constructor('c',None if case=='no_device_kwarg' else 'cpu' if case=='string_device' else torch.device('cpu'));constructors=[c];source=c
    if case in ['two_constructors','linked_output_blocked']:
        d=constructor('d');constructors.append(d);source=g.call_function(a.add.Tensor,(c,d));source.meta['val']=val([2],'cpu')
    elif case=='repeated_input':source=g.call_function(a.add.Tensor,(c,c));source.meta['val']=val([2],'cpu')
    elif case=='cpu_input_blocked':
        cpu=g.placeholder('cpu_input');cpu.meta['val']=val([2],'cpu');source=g.call_function(a.add.Tensor,(c,cpu));source.meta['val']=val([2],'cpu')
    elif case=='opaque_blocked':source=g.call_function(opaque,(c,));source.meta['val']=val([2],'cpu')
    if case=='gpu_boundary_two_inputs':
        index=constructor('index',dtype=torch.int64);constructors.append(index);result=g.call_function(a.index_put.default,(p,[index],c))
    else:result=g.call_function(a.copy.default,(p,source))
    result.meta['val']=val([2],'cpu' if case=='no_target' else 'cuda:0');g.output((result,c) if case in ['output_blocked','output_allowed','linked_output_blocked'] else (result,));gm=torch.fx.GraphModule({},g)
    mover=ConstructorMoverPass('cuda',allow_outputs=case=='output_allowed');cpu_counts={n.name:v for n,v in mover.get_cpu_indeg_count(g).items()};before=str(g);ids=[id(n) for n in g.nodes];old_meta=[c.meta['val'] for c in constructors]
    mover(g);g.lint();assert ids==[id(n) for n in g.nodes] and all(c.meta['val'] is m for c,m in zip(constructors,old_meta))
    after=[str(c.kwargs.get('device','absent')) for c in constructors]
    report[case]={'constructor_kwargs_after':after,'metadata_devices_after':[str(c.meta['val'].device) for c in constructors],'cpu_indeg_before':cpu_counts,'identity_and_metadata_unchanged':True,'execution':'metadata-only FX pass; no GPU allocation or collective'}
    (OUT/f'{case}_before.txt').write_text(before+'\n');(OUT/f'{case}_after.txt').write_text(str(g)+'\n')
for c in ['movable','output_allowed','two_constructors','gpu_boundary_two_inputs']:assert all(d=='cuda:0' for d in report[c]['constructor_kwargs_after'])
for c in ['output_blocked','no_target','multiple_targets','cpu_input_blocked','linked_output_blocked','repeated_input','opaque_blocked']:assert all(d=='cpu' for d in report[c]['constructor_kwargs_after'])
assert report['no_device_kwarg']['constructor_kwargs_after']==['absent'] and report['string_device']['constructor_kwargs_after']==['cpu']
assert report['repeated_input']['cpu_indeg_before']['add_tensor']==2
# 特殊 index_put 保护条件依照实际 dtype 列表，而不是只按源码注释中的 byte 一词。
for name,dtype,shape,device in [('bool_scalar',torch.bool,[],'cpu'),('int8_scalar',torch.int8,[],'cpu'),('uint8_scalar',torch.uint8,[],'cpu'),('int64_scalar',torch.int64,[],'cpu'),('bool_singleton',torch.bool,[1],'cpu'),('bool_vector',torch.bool,[2],'cpu'),('bool_gpu_value',torch.bool,[],'cuda:0')]:
    g=torch.fx.Graph();p=g.placeholder('p');p.meta['val']=val([2],'cuda:0');mask=g.placeholder('mask');mask.meta['val']=val([2],'cuda:0',dtype)
    v=g.call_function(a.full.default,(shape,3.),{'dtype':torch.float32,'device':torch.device(device)});v.meta['val']=val(shape,device)
    n=g.call_function(a.index_put.default,(p,[mask,None],v));n.meta['val']=val([2],'cuda:0');g.output((n,));gm=torch.fx.GraphModule({},g)
    guarded=is_index_put_and_requires_h2d_sync_for_gpu_value(n);ConstructorMoverPass('cuda')(g);g.lint()
    report[name]={'guarded':guarded,'value_device_kwarg_after':str(v.kwargs['device']),'input_dtype':str(dtype),'value_shape':shape,'scope':'source predicate and Fake metadata pass only; not ATen index execution validity'}
assert all(report[c]['guarded'] for c in ['bool_scalar','int8_scalar','bool_singleton'])
assert not any(report[c]['guarded'] for c in ['uint8_scalar','int64_scalar','bool_vector','bool_gpu_value'])
assert report['bool_scalar']['value_device_kwarg_after']=='cpu' and report['uint8_scalar']['value_device_kwarg_after']=='cuda:0'
try:ConstructorMoverPass(torch.device('cuda'))
except AssertionError as e:report['invalid_target']={'error':type(e).__name__,'message':str(e)}
else:raise AssertionError('target type must be str')
g=torch.fx.Graph();n=g.placeholder('n');n.meta['val']=(val([2],'cuda:0'),);m=ConstructorMoverPass('cuda');assert m.get_node_device(n) is None
report['tuple_metadata']={'device':None,'scope':'get_node_device accepts a Tensor, not arbitrary nested Tensor collections'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
