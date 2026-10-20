"""原生CPU缓存冷/暖/条件变化/并发，另用显式fixture检查CUDA/ROCm键协议。"""
import json,os,sys,tempfile,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/source-binary-cache';OUT.mkdir(exist_ok=True)
SOURCE='''#ifndef LEARNING_INCREMENT
#define LEARNING_INCREMENT 7
#endif
extern "C" int kernel(int x) { return x + LEARNING_INCREMENT; }
'''
def child(label,cache,flags,gate=''):
    os.environ['TORCHINDUCTOR_CACHE_DIR']=cache
    os.environ['TORCH_WARM_POOL']='0'
    import ctypes
    from unittest.mock import patch
    import torch._inductor.codecache as cc
    from torch._inductor.cpp_builder import CppBuilder
    calls=[];original=CppBuilder.build
    def build(self):
        calls.append(self.get_command_line());return original(self)
    if gate:
        (OUT/(label+'.ready')).touch()
        deadline=time.monotonic()+60
        while not Path(gate).exists():
            if time.monotonic()>deadline:raise TimeoutError('controller gate')
            time.sleep(.02)
    with patch.object(CppBuilder,'build',build):
        get_result=cc.CppCodeCache.load_async(SOURCE,extra_flags=flags)
        before=len(calls)
        lib=get_result();second=cc.CppCodeCache.load_async(SOURCE,extra_flags=flags)
        assert second is get_result and second() is lib
        lib.kernel.argtypes=[ctypes.c_int];lib.kernel.restype=ctypes.c_int
        result=lib.kernel(5)
    binary=Path(lib._name);key=lib.key
    source_path=Path(cc.get_path(key,'cpp')[2])
    row={'label':label,'key':key,'source_path':str(source_path),'binary_path':str(binary),'builds_before_get_result':before,'build_calls':sum(str(binary) in command for command in calls),'all_build_calls_including_isa_probes':len(calls),'commands':calls,'same_getter':second is get_result,'same_library':second() is lib,'value':result,'source_exists':source_path.exists(),'binary_exists':binary.exists(),'flags':flags}
    (OUT/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    return row
if len(sys.argv)>1 and sys.argv[1]=='--child':
    child(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),sys.argv[5] if len(sys.argv)>5 else '')
    raise SystemExit
run_dir=Path(tempfile.mkdtemp(prefix='native-',dir=OUT));cache=run_dir/'cache'
env=dict(os.environ,PYTHONPATH='/work/pytorch',TORCH_WARM_POOL='0')
def command(label,cache,flags,gate=''):
    return [sys.executable,__file__,'--child',label,str(cache),json.dumps(flags),gate]
for label,flags in [('cold',[]),('warm',[]),('variant',['-DLEARNING_INCREMENT=9'])]:
    subprocess.run(command(label,cache,flags),env=env,check=True,timeout=120)
gate=run_dir/'start';race_cache=run_dir/'race-cache'
for label in ('race_a','race_b'):(OUT/(label+'.ready')).unlink(missing_ok=True)
procs=[subprocess.Popen(command(label,race_cache,[],str(gate)),env=env) for label in ('race_a','race_b')]
try:
    deadline=time.monotonic()+60
    while not all((OUT/(label+'.ready')).exists() for label in ('race_a','race_b')):
        assert all(p.poll() is None for p in procs)
        if time.monotonic()>deadline:raise TimeoutError('child initialization')
        time.sleep(.02)
    gate.touch()
    for p in procs:assert p.wait(timeout=120)==0
finally:
    for p in procs:
        if p.poll() is None:p.terminate();p.wait(timeout=30)
rows={label:json.loads((OUT/(label+'.json')).read_text()) for label in ('cold','warm','variant','race_a','race_b')}
assert rows['cold']['build_calls']==1 and rows['warm']['build_calls']==0
assert rows['cold']['key']==rows['warm']['key']!=rows['variant']['key']
assert rows['variant']['build_calls']==1 and rows['variant']['value']==14
assert sum(rows[k]['build_calls'] for k in ('race_a','race_b'))==1
assert rows['race_a']['key']==rows['race_b']['key']==rows['cold']['key']
assert all(r['value']==12 for k,r in rows.items() if k!='variant')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(run_dir/'protocol-cache')
os.environ['TORCH_WARM_POOL']='0'
from unittest.mock import patch
import torch._inductor.codecache as cc
keys={}
padded=' \n  learning-cache-text\n '
k,p=cc.write(padded,'txt');k2,p2=cc.write('learning-cache-text','txt')
assert (k,p)==(k2,p2) and Path(p).read_text()==padded
k3,p3=cc.write('learning-cache-text','py');assert k3==k and p3!=p
k4,p4=cc.write('learning-cache-text','txt',extra='target=other');assert k4!=k
k5,p5=cc.write('learning-cache-text','txt',specified_dir='custom');assert k5==k and Path(p5).parent.name=='custom'
assert cc.get_hash(b'abc',extra='one',hash_type='spv')==cc.get_hash(b'abc',extra='two',hash_type='spv')
keys={'strip_same_key':True,'first_original_content_preserved':True,'extension_changes_path_not_key':True,'extra_changes_code_key':True,'directory_changes_path_not_key':True,'spv_extra_ignored':True}
backends={}
# 实际compile逻辑配合假编译命令/假产物，不调用nvcc/hipcc，不加载文件。
for name,cls,symbol in [('cuda',cc.CUDACodeCache,'cuda_compile_command'),('rocm',cc.ROCmCodeCache,'rocm_compile_command')]:
    cls.cache_clear();commands=[]
    def command_fixture(inputs,output,ext,extra_args=None):
        return 'fixture-compiler '+str(output)+' --target='+ext+' '+','.join(extra_args or [])
    def fake_compile(parts,**kwargs):
        commands.append(parts);Path(parts[1]).write_text('NOT A BINARY: fixture only');return '' if kwargs.get('text') else b''
    version_patch=patch.object(cc.ROCmCodeCache,'_logged_compiler_version',True)
    with patch.object(cc,symbol,side_effect=command_fixture),patch.object(cc.subprocess,'check_output',side_effect=fake_compile),version_patch:
        first=cls.compile('fixture '+name,'so',extra_args=['-DVALUE=1'])
        second=cls.compile('fixture '+name,'so',extra_args=['-DVALUE=2'])
        cls.cache_clear()
        third=cls.compile('fixture '+name,'so',extra_args=['-DVALUE=3'])
        other=cls.write('fixture '+name,'o')
        assert first==second==third and len(commands)==1 and other[0]!=first[1]
    backends[name]={'fixtures_only':True,'extra_args_same_key':True,'memory_then_disk_skip':True,'compiler_calls':len(commands),'changed_extension_changes_key':True,'first_command':commands[0]}
report={'native':rows,'keys':keys,'device_protocol':backends}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'native_builds':{k:r['build_calls'] for k,r in rows.items()},'values':{k:r['value'] for k,r in rows.items()},'keys':keys,'device_protocol':backends},ensure_ascii=False,indent=2))
