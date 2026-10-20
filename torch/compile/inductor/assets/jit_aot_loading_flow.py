"""JIT返回当前入口；AOT返回产物，另行装载后才能运行。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1740" viewBox="0 0 1400 1740"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1400" height="1740" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,v,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(v)}</text>')
def b(x,y,w,h,title,lines,c):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(x+w/2,y+36,title,24)
 for i,v in enumerate(lines):t(x+w/2,y+80+i*34,v)
def e(d,dash=False):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" '+('stroke-dasharray="9 6" ' if dash else '')+'marker-end="url(#a)"/>')
t(700,45,'JIT 与 AOT 装载：当前进程入口、磁盘产物、运行容器各是什么',28)
b(140,90,1120,120,'输入：GraphLowering 的生成程序与常量状态',['compile_to_fn 根据 aot_mode 选择出口'],('#f2f2f2','#888'))
e('M700 210 V245 H355 V290');e('M700 245 H1045 V290');t(200,273,'JIT',20);t(880,273,'AOT（要求 C++ wrapper）',20)
b(60,290,590,250,'① JIT 模块与入口',['生成 wrapper 源码 → 写 Python 文件','PyCodeCache：执行模块顶层 / 查询模块缓存','顶层可包含 kernel 编译与 wait','冷加载执行后安装常量 attrs、保存行映射','Graph 保存 key / path / linemap'],('#e1f2ff','#368dc0'))
b(750,290,590,250,'② AOT 产物构建',['生成 C++ wrapper 源码','编译 wrapper 与常量产物、链接','可另带设备二进制 / 外部节点描述','普通分支得到 .so；package 分支归档','编译出口尚未建立调用者的 runner'],('#e5f6e5','#40a147'))
e('M355 540 V600');e('M1045 540 V600')
b(60,600,590,140,'JIT 编译出口：module.call',['当前 Python 进程可调用对象','交给 CompiledFxGraph 等调用封装'],('#fff5ce','#d6a000'))
b(750,600,590,140,'AOT 编译出口：产物路径字符串',['尚未执行模型','归档需先还原 / 构建 .so，再按运行协议装载'],('#fff5ce','#d6a000'))
e('M1045 740 V765 H1310 V810 H1045 V835',True);t(1045,795,'后续由调用者另外发起装载',19)
b(750,835,590,245,'③ AOT 容器装载与运行',['动态库查找 AOTInductorModelContainer* 符号','CreateWithDevice → 建立模型容器','runner 收到 Tensor 列表 → 转换句柄','Run 传入 stream / proxy executor','取回输出句柄，转成 Tensor'],('#f0e5ff','#9959c6'))
e('M355 740 V810 H110 V1100 H355 V1150',True);t(355,945,'后续模型调用',21);t(355,982,'不是模块 import 时执行 Tensor 计算',19)
b(60,1150,590,160,'运行 JIT 入口',['call(args) 读取常量、分配输出、启动 kernel','具体模型输入在这个时点交入','得到本次输出 Tensor'],('#e1f2ff','#368dc0'))
e('M1045 1080 V1150')
b(750,1150,590,160,'运行 AOT 入口',['低层 runner.run(flat_tensors)','高层适配还根据 call_spec 展平 / 重建结构','得到本次输出 Tensor'],('#f0e5ff','#9959c6'))
b(140,1400,1120,135,'共享：源码 / 二进制缓存、任务等待、设备 wrapper 与 runtime',['本图按装载出口区分时点；共享机制仍由各自模块解释','虚线标记后续调用时点，不代表编译出口已经执行模型'],('#f2f2f2','#888'))
t(700,1640,'三个直接子模块为①②③；新设备需同时满足编译产物、容器 ABI、设备解析和 stream/句柄协议。',19)
s.append('</g></svg>');svg=''.join(s);(P/'jit-aot-loading.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'jit-aot-loading.png'))
im=Image.open(P/'jit-aot-loading.png').convert('RGB');im.thumbnail((950,1200));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'jit-aot-loading-review.png')
