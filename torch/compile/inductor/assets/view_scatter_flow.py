"""View/scatter 的中间表示与实际分解分支。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
H=1420
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{H}" viewBox="0 0 1200 {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1200" height="{H}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(x,y,w,h,title,lines,fill,stroke):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+35,title,24)
 for j,s in enumerate(lines):t(x+w/2,y+78+j*30,s,20)
def e(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.8" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
t(600,43,'View/scatter：先合并更新路径，再按选定语义展开',29)
t(600,94,'输入：有 metadata 的 FX 图；最终输出：普通 ATen 操作组成的同一张图',21)
a.append('<rect x="35" y="120" width="1130" height="1210" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(160,155,880,155,'canonicalize_view_scatter_ops：扫描并记录视图路径',['保存 node→base 与 node→ViewOp 列表','普通 scatter → 一步广义 scatter；可对接的嵌套更新 → 路径合并','内层结果仍被使用时，改接为合并结果的对应 view'],'#e3f3ff','#368dc0')
b(160,365,880,125,'广义 scatter：inp、src、view_ops',['功能式：clone base 后沿路径 copy；原地式：直接写 base','graph_call_function 在创建节点时计算 FakeTensor metadata'],'#e5f6e5','#40a147')
b(160,545,880,125,'外部协作：存储可写性与原地化核心',['可写性判断结合 should_reinplace_scatter 的策略','允许复用时把功能 target 改为原地 target；否则保留'],'#f4f4f4','#888')
b(160,725,880,125,'decompose_generalized_scatter：读取已选定的 target',['原地 target，或路径含 as_strided / diagonal → 写入分解','其余功能 target → 递归回填分解；这里不再做可写性判断'],'#fff3c9','#dca300')
b(65,965,500,165,'功能式分解',['沿外层 view 递归进入目标区域','最内层 scatter 更新，再逐层 scatter 回填','结果是新的 base；不修改原输入'],'#e3f3ff','#368dc0')
b(635,965,500,165,'写入分解',['功能 target：先 clone；原地 target：不 clone','按 view_ops 顺序找目标，最后 copy_(src)','返回完整 base，不把最后一个 view 当结果'],'#ffe4e4','#ef5555')
b(160,1210,880,95,'重接广义节点的使用者，再删除旧节点',['正常结束返回 None；新节点带 val，随后才进行最终 recompile'],'#f0e5ff','#9959c6')
for y,z in [(310,365),(490,545),(670,725)]:e(f'M600 {y} V{z}')
e('M600 850 V900 H315 V965');e('M600 900 H885 V965');t(315,938,'功能 target 且不强制 mutation',18);t(885,938,'原地 target 或强制 mutation',18)
e('M315 1130 V1170 H440 V1210');e('M885 1130 V1170 H760 V1210')
t(600,1375,'实线：改图期间交接。灰框归兄弟模块；广义节点是临时表示，不是最终可生成的 forward',19)
a.append('</g></svg>');svg=''.join(a);(P/'view-scatter.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'view-scatter.png'))
im=Image.open(P/'view-scatter.png').convert('RGB');im.thumbnail((800,1000));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'view-scatter-review.png')
