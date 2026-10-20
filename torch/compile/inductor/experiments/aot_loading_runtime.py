"""在另一个进程中只装载已有AOT产物，不重新编译。"""
import json
from pathlib import Path
import torch
from torch._export import aot_load
O=Path(__file__).resolve().parent/'jit-aot-loading'
aot=json.loads((O/'aot.json').read_text());assert aot['compiled'] and aot['loaded']
path=aot['path'];runner=torch._C._aoti.AOTIModelContainerRunnerCpu(path,1)
names=runner.get_constant_names_to_original_fqns()
optimized=aot_load(path,'cpu');runs=[]
for shift in [0.,1.5]:
    x=torch.arange(6,dtype=torch.float32).reshape(2,3)+shift;before=x.clone()
    actual=optimized(x);assert isinstance(actual,tuple) and len(actual)==1
    torch.testing.assert_close(actual[0],x+torch.tensor([10.,20.,30.]))
    assert torch.equal(x,before) and actual[0].untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
    runs.append({'shift':shift,'output':actual[0].tolist(),'input_preserved':True,'output_aliases_input':False,'output_type':type(actual).__name__})
assert runs[0]['output']!=runs[1]['output']
report={'compiled_in_this_process':False,'path':path,'constant_names_to_fqns':names,'runs':runs,'call_spec':list(runner.get_call_spec())}
(O/'aot-runtime.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
