"""post_compile选择适配入口，调用时才处理当前输入或延迟捕获。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="2000" viewBox="0 0 1450 2000"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1450" height="2000" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,v,z=22):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(v)}</text>')
def b(y,h,title,lines,c):
 s.append(f'<rect x="70" y="{y}" width="1110" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(625,y+36,title,25)
 for i,v in enumerate(lines):t(625,y+78+i*33,v)
def e(d,dash=False):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" '+('stroke-dasharray="9 6" ' if dash else '')+'marker-end="url(#a)"/>')
t(725,45,'调用适配与 CUDA Graph：先安装当前入口，再使用本次输入运行',28)
b(90,115,'输入：CompiledFxGraph + 示例输入 + 共享 BoxedBool',['命中恢复 / 新编译 / bypass 汇合；AOT 路径字符串直接返回，不走本层'],('#f2f2f2','#888'))
b(260,160,'① 编译后策略与运行状态',['将 output_strides 回填当前 TracingContext','已有禁用原因 / 条件不满足 → 共享 CUDA Graph 开关变 False','允许捕获 → 交③安装延迟入口；此时不等于已经捕获'],('#e1f2ff','#368dc0'))
b(475,150,'③ 捕获、重放与 Graph Trees：安装延迟 callable',['保存原 callable、static 索引、设备、常量、mutation 等信息','返回包装入口；首次真实调用时才触发捕获实现'],('#f0e5ff','#9959c6'))
b(695,160,'② 输入对齐与调用转交：安装普通运行适配',['CUDA Graph 已禁用：按 inputs_to_check 决定是否包装 callable','无检查项 → 原入口；有检查项 → 运行时先检查实际地址','允许 CUDA Graph 的路径：对齐由③中的相应入口处理'],('#e5f6e5','#40a147'))
b(925,125,'输出：原 CompiledFxGraph，current_callable 已按条件更新',['post_compile 返回对象；编译后适配不写回 FX 缓存描述'],('#fff4ce','#d6a000'))
b(1130,155,'之后真实调用：CompiledFxGraph.__call__(inputs_list)',['普通路径经②：必要时替换列表中的 Tensor，再转交编译入口','Graph 路径经③：延迟建立 / 查找运行方案，再转交捕获或重放','参数列表可被消费；不能把它与 Tensor 存储修改混为一谈'],('#fff4ce','#d6a000'))
b(1345,185,'③运行阶段：捕获与重放有自己的状态和失败边界',['legacy：固定输入/输出存储、复制动态输入、重放','Trees：整型键、执行路径、存储活性、静态地址与 mutation 条件','不适用可选择普通执行；捕获异常不保证自动兜底','GPU 捕获本轮未运行，协议 fixture 与真实 CPU 证据分开'],('#f0e5ff','#9959c6'))
b(1600,130,'交付本次结果与状态变化',['输出 / alias / mutation 由真实运行入口决定','新设备需明确对齐、固定地址、allocator、stream 和生命周期协议'],('#f2f2f2','#888'))
for a,z in [(205,260),(420,475),(855,925),(1285,1345),(1530,1600)]:e(f'M625 {a} V{z}')
e('M625 625 V695');t(880,665,'安装后继续 post_compile',18)
e('M1180 340 H1240 V775 H1180');s.append('<text x="1260" y="510" font-size="18">禁用：</text><text x="1260" y="540" font-size="18">跳过③安装</text>')
e('M625 1050 V1070 H110 V1110 H625 V1130',True);t(625,1095,'后续调用时点',19)
e('M1180 1210 H1240 V1665 H1180');s.append('<text x="1260" y="1400" font-size="18">普通入口：</text><text x="1260" y="1430" font-size="18">跳过③运行</text>');t(780,1320,'Graph 路径',18)
t(725,1840,'①②③是三个直接职责；普通非 Graph 执行不需要经过③的捕获/重放运行阶段。',20)
t(725,1890,'实线表示当前阶段的交接，虚线表示返回之后的调用时点；具体分支条件写在节点中。',19)
s.append('</g></svg>');svg=''.join(s);(P/'call-adaptation.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'call-adaptation.png'))
im=Image.open(P/'call-adaptation.png').convert('RGB');im.thumbnail((900,1280));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'call-adaptation-review.png')
