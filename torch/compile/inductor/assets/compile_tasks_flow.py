"""编译任务的提交与结果消费；颜色和灰度均可读。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="1620" viewBox="0 0 1250 1620"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1250" height="1620" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,text,z=22):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def b(y,h,title,lines,c):
 s.append(f'<rect x="70" y="{y}" width="1000" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(570,y+35,title,25)
 for i,line in enumerate(lines):t(570,y+75+i*31,line)
def e(d,dash=False):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" '+('stroke-dasharray="9 6" ' if dash else '')+'marker-end="url(#a)"/>')
t(625,44,'编译任务与等待：源码 → 延迟任务 / 立即对象 → scope 中的可用入口',27)
b(90,110,'输入：生成模块初始化期间的 kernel 编译声明',['例如 cpp_pybinding(argtypes, source) 或 triton(name, source)'],('#f2f2f2','#888'))
b(250,165,'① 编译入口与后端任务',['CPU C++ / Halide：组织缓存加载，返回立即入口或 LambdaFuture','CUDA / ROCm C++：submit 线程任务；Triton：看进程池是否就绪','线程数 ≤ 1 时，相应编译入口走同步分支'],('#e1f2ff','#368dc0'))
b(475,180,'② 线程池、进程池与 worker：只对相应异步路径',['submit → 同进程线程池；Triton 就绪路径 → 编译进程池','C++ worker 形成二进制；Triton worker 只 warm_cache_only 预编译','返回 Future；任务可与其余模块初始化交错，不在此执行模型'],('#e5f6e5','#40a147'))
b(725,140,'生成模块保存任务，再调用 wait(scope)',['scope 中混有普通对象、Future、CodeCacheFuture','①的立即结果同样放入 scope；它不需要解析 Future'],('#fff6cf','#d6a000'))
b(925,185,'③ Future 解析与 scope 交接：需要入口时消费',['线程数 > 1：逐个取顶层 Future.result()，替换同名 scope 项','C++：等待编译、按需装载；Triton：先等 worker，再主进程 precompile','失败立即抛出；已替换的前项保留，后项还未访问'],('#f0e5ff','#9959c6'))
b(1175,130,'输出：当前模块 scope 中的后端对象 / 可调用入口',['wait 本身返回 None；不会递归处理容器内的 Future','线程数 ≤ 1：wait 不解析 Future，正常入口已同步完成'],('#f0e5ff','#9959c6'))
b(1370,110,'之后调用模型：wrapper 才使用这些入口执行张量计算',['新设备需要另外提供相应编译器、装载及 runtime launch 实现'],('#f2f2f2','#888'))
for a,z in [(200,250),(415,475),(865,925),(1110,1175),(1305,1370)]:e(f'M570 {a} V{z}')
e('M570 655 V725',True);t(770,695,'提交后保存任务；虚线不是阻塞等待',17)
e('M1070 335 H1135 V795 H1070');s.append('<text x="1150" y="520" font-size="18">同步 /</text><text x="1150" y="550" font-size="18">未就绪</text><text x="1150" y="580" font-size="18">立即结果</text>')
t(625,1555,'①②③对应三个直接子模块。共享源码/二进制缓存服务①与③，在相邻缓存篇展开。',19)
s.append('</g></svg>');svg=''.join(s);(P/'compile-tasks.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'compile-tasks.png'))
im=Image.open(P/'compile-tasks.png').convert('RGB');im.thumbnail((800,1100));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'compile-tasks-review.png')
