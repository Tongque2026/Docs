"""元素 handler 的逐操作分派；三个职责交错参与，不画成三个串行编译阶段。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1560" viewBox="0 0 1400 1560"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1400" height="1560" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,text,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(x,y,w,h,title,lines,color,z=21):
 fill,stroke=color;s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+36,title,24)
 for i,line in enumerate(lines):t(x+w/2,y+77+i*31,line,z)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(700,44,'元素表达式与目标代码：每个操作经 handler 形成源码与状态',29)
box(130,90,1140,130,'共享：后端与循环组织',['交入 LoopBody、循环索引、范围、布局与当前 Kernel','不是传入一批真实 Tensor 元素，返回值也不是模型数值'],('#f2f2f2','#888'))
box(130,280,1140,160,'① ops 解释与类型语义',['LoopBody 内部 FX 图在当前 ops handler 下逐节点解释','OpsWrapper / OpsValue 保留编译期运算接口；目标 overrides 与 dtype 信息参与生成','同一个动作可能同时用到下面多个职责'],('#e3f3ff','#368dc0'))
e('M700 220 V280');e('M700 440 V495 H250 V550');e('M700 495 V550');e('M700 495 H1150 V550')
box(50,550,400,225,'普通运算：① → ③',['① 目标 overrides 返回表达式','③ CSE 查表达式缓存、命名','首次写 compute；命中复用临时值','返回 CSEVariable 给解释器'],('#e3f3ff','#368dc0'),19)
box(500,550,400,225,'加载：③ → ② → ③',['③ 先查写后读缓存','命中：直接返回已有临时值','未命中：② 参数 / 地址 / mask','③ 记录加载代码和临时值'],('#e5f6e5','#40a147'),19)
box(950,550,400,225,'写出：③ → ② / 跳过',['③ 登记写缓存与 buffer 名称','② 发出目标 store / atomic 等','已移除 buffer：可跳过实际写出','保留后续失效与生命周期所需状态'],('#fff0cf','#d8a000'),19)
s.append('<path d="M250 775 V840 H1150 V775 M700 775 V840" fill="none" stroke="#222" stroke-width="3"/>');e('M700 840 V900')
box(130,900,1140,195,'③ 临时值复用与生成区：持续积累状态',['上述分派对 LoopBody 各节点重复；需要时将临时值交回解释器','loads / compute / stores、参数使用、store cache、must_keep 等状态随之改变','作用域结束要处理缓存失效与局部 buffer；不能把复用当全局内存缓存','共享 exp 案例：buf0 → tmp1 转发，最终只保留两个图输出写出'],('#f0e5ff','#9959c6'))
box(130,1160,1140,165,'共享：循环 / 函数封装 → wrapper → 编译加载',['CPU 将生成区插入循环；Triton 还组织块、mask 与归约作用域','本层交付源码片段与参数状态；后续生成完整入口才接收真实输入','当前 CPU 低精度往返案例出现中间舍入差异，正文单独保留对照'],('#f2f2f2','#888'))
e('M700 1095 V1160')
t(700,1400,'三个直接子模块：① ops 解释与类型语义　② 参数索引与访存　③ 临时值复用与生成区',21)
t(700,1460,'实线：生成期分派与状态交接；三条操作路径不是整段循环依次经过的三个阶段',20)
t(700,1510,'CPU：实际编译执行。Triton：表达式 / 元信息验证。TextKernel：明确的文本协议实验',20)
s.append('</g></svg>');svg=''.join(s);(P/'element-codegen.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'element-codegen.png'))
im=Image.open(P/'element-codegen.png').convert('RGB');im.thumbnail((950,1100));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'element-codegen-review.png')
