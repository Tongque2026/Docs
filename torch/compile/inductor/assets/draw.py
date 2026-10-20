from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1470" viewBox="0 0 1000 1470"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1000" height="1470" fill="white"/><g font-family="Noto Sans CJK SC, sans-serif" fill="#202020">']
def text(x,y,lines,size=22,weight='normal'):
    for i,line in enumerate(lines): parts.append(f'<text x="{x}" y="{y+i*32}" text-anchor="middle" font-size="{size}" font-weight="{weight}">{escape(line)}</text>')
def box(x,y,w,h,fill,stroke,lines):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');text(x+w/2,y+36,lines)
def arrow(x,y,x2,y2,dash=False):parts.append(f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#arrow)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
text(500,42,['Inductor：从一张 FX 图到可调用程序'],30,'bold')
box(225,75,550,100,'#fafafa','#555',['输入：GraphModule + example_inputs','一张 forward / backward / inference 图'])
parts.append('<rect x="35" y="215" width="930" height="950" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
text(160,253,['Inductor 内部'],24,'bold')
box(205,280,590,105,'#e3f3ff','#368dc0',['① 图准备','形状传播、Post-grad Passes → 整理后的 FX 图'])
box(205,440,590,120,'#e5f6e5','#40a147',['② Lowering','GraphLowering.run → 计算、布局、存储的 IR','状态保存在 GraphLowering 对象上'])
box(205,620,590,105,'#fff3c9','#dca300',['③ 调度与融合','Scheduler → 依赖、融合分组、最后使用位置'])
box(205,810,590,145,'#ffe4e4','#ef5555',['④ 代码生成与运行封装','设备 kernel / 外部调用 + wrapper','编译、加载 → module.call'])
box(205,1000,590,115,'#f0e5ff','#9959c6',['运行封装的出口','CompiledFxGraph(current_callable, 元数据)','缓存恢复后同样完成运行适配'])
arrow(500,175,500,280);arrow(500,385,500,440);arrow(500,560,500,620);arrow(500,725,500,810)
text(855,750,['Scheduler.codegen','驱动后端生成'],16)
arrow(500,955,500,1000)
parts.append('<path d="M500 195 H110 V1055 H205" fill="none" stroke="#555" stroke-width="2.5" stroke-dasharray="8 6" marker-end="url(#arrow)"/>')
text(840,1195,['虚线：缓存命中','跳过重新编译'],16)
text(855,520,['共享支撑','FX 图缓存','入口先查','出口保存'],18)
box(225,1230,550,85,'#efe2ff','#9752c5',['返回：可调用程序（尚非计算结果）'])
arrow(500,1115,500,1230)
box(225,1370,550,70,'#fafafa','#555',['之后调用：真实输入 → 输出张量 / 输入更新'])
arrow(500,1315,500,1370,True)
parts.append('</g></svg>')
svg=''.join(parts);(P/'inductor-flow.svg').write_text(svg)
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'inductor-flow.png'))
