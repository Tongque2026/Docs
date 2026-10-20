"""从单图编译入口采集 CPU 实验；缓存全部写入独立目录。"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT / 'experiments/cache')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
import json
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import config, metrics
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.graph import GraphLowering

report = {'torch_version': torch.__version__, 'torch_file': torch.__file__, 'cases': {}}
original = GraphLowering.save_output_code
original_run = GraphLowering.run
def observed_run(self, *args):
    result = original_run(self, *args)
    (ROOT / "experiments" / (active_name + "_lowered.txt")).write_text(str(self.module.graph) + "\n\n" + "\n".join(repr(op) for op in self.operations))
    return result
GraphLowering.run = observed_run
try:
    with config.patch({'fx_graph_cache': False, 'force_disable_caches': True, 'triton.cudagraphs': False}):
        for name, fn in [('pointwise', lambda x,y: (torch.relu(x+y)*2,)), ('alias', lambda x,y: (x.view(6),))]:
            active_name = name
            x = torch.tensor([[-2.,-1.,0.],[1.,2.,3.]])
            y = torch.ones_like(x)
            before = x.clone()
            gm = make_fx(fn)(x,y)
            (ROOT / 'experiments' / (name + '_fx.txt')).write_text(str(gm.graph))
            GraphLowering.save_output_code = staticmethod(lambda code, name=name: (ROOT / 'experiments' / (name+'_output.py')).write_text(code))
            metrics.reset()
            compiled = compile_fx_inner(gm, [x,y], is_inference=True)
            args = [x,y]
            result = compiled(args)
            expected = fn(x,y)
            torch.testing.assert_close(result, expected)
            report['cases'][name] = {'callable_type': type(compiled).__name__, 'boxed_call': compiled._boxed_call, 'output': result[0].tolist(), 'shape': list(result[0].shape), 'stride': list(result[0].stride()), 'dtype': str(result[0].dtype), 'device': str(result[0].device), 'input_unchanged': torch.equal(x,before), 'output_aliases_x': result[0].untyped_storage().data_ptr()==x.untyped_storage().data_ptr(), 'generated_kernel_count': metrics.generated_kernel_count, 'args_length_after_call': len(args)}
finally:
    GraphLowering.save_output_code = staticmethod(original)
    GraphLowering.run = original_run
(ROOT / 'experiments/results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(report, indent=2, ensure_ascii=False))
