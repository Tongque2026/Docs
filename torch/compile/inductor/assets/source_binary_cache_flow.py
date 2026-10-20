"""源码/条件形成键，锁中生成产物，按后端协议交付。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1810" viewBox="0 0 1400 1810"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1400" height="1810" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,v,z=22):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(v)}</text>')
def b(y,h,title,lines,c):
 s.append(f'<rect x="80" y="{y}" width="1090" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(625,y+36,title,25)
 for i,v in enumerate(lines):t(625,y+78+i*32,v)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(700,45,'源码和二进制缓存：同一个键，区分文件存在与当前进程已装载',28)
b(90,115,'输入：源码 + 后端 / 编译条件 + 期望产物类型',['调用者来自 AsyncCompile、生成模块加载，或 AOT 编译'],('#f2f2f2','#888'))
b(260,150,'② 编译条件与锁内构建：先准备键所需的条件',['CPU C++：占位路径编译命令含 ISA / extra_flags 等','CUDA / ROCm：各自生成编译命令；条件覆盖以实际 write 调用为准'],('#e4f5e4','#40a147'))
b(465,160,'① 缓存键、路径与原子写入',['write 以 content.strip() 与 extra 等生成 key，扩展名和目录形成路径','源文件不存在 → 同目录临时写入后 rename；存在 → 复用','路径命中不表示已得到当前进程函数'],('#e1f2ff','#368dc0'))
b(680,150,'③ 后端产物与进程内复用：检查已有记录',['C++：key → 取得库的闭包，闭包可保存已装载 lib','CUDA / ROCm：key → input/output 路径；不是已打开 DLL'],('#f0e5ff','#9959c6'))
b(890,175,'② 编译条件与锁内构建：需要时检查磁盘产物',['CPU 示例：持有 key.lock 后再次检查二进制','已存在 → 复用；缺失 → 调用对应编译器形成产物','并发等待者拿到锁后再检查，避免重复构建同一产物'],('#e4f5e4','#40a147'))
b(1125,170,'③ 后端产物与进程内复用：按调用协议取结果',['C++：等待已提交任务，再确认产物，当前进程装载并保留 lib','CUDA / ROCm compile → 路径元组；load → 新 DLLWrapper','Python 模块 / Triton 描述与 Halide 产物有各自缓存协议'],('#f0e5ff','#9959c6'))
b(1370,140,'输出：源码路径 / 二进制路径，或当前进程可用对象',['交还 AsyncCompile / JIT-AOT 装载；不返回模型的 Tensor 答案','模型计算由之后的 wrapper 调用执行'],('#fff4cf','#d6a000'))
for a,z in [(205,260),(410,465),(625,680),(830,890),(1065,1125),(1295,1370)]:e(f'M625 {a} V{z}')
e('M1170 755 H1230 V1210 H1170');s.append('<text x="1250" y="925" font-size="18">已装载 /</text><text x="1250" y="955" font-size="18">已完成记录：</text><text x="1250" y="985" font-size="18">按后端</text><text x="1250" y="1015" font-size="18">取结果</text>')
b(1575,115,'边界：三个职责会被反复调用，不是只执行一次的三个阶段',['共享任务池由“编译任务与等待”拥有；JIT / AOT 最终出口由相邻模块拥有'],('#f2f2f2','#888'))
t(700,1760,'CPU 锁与装载主线已真实执行；GPU 路径解释来自源码及明确 fixture，不表示设备编译已运行。',19)
s.append('</g></svg>');svg=''.join(s);(P/'source-binary-cache.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'source-binary-cache.png'))
im=Image.open(P/'source-binary-cache.png').convert('RGB');im.thumbnail((850,1200));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'source-binary-cache-review.png')
