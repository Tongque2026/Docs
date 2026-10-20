"""批量与通信的三个职责及真实调用位置。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
H=1290
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{H}" viewBox="0 0 1100 {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="{H}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(y,h,title,lines,fill,stroke):
 a.append(f'<rect x="85" y="{y}" width="780" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(475,y+35,title,24)
 for j,s in enumerate(lines):t(475,y+78+j*31,s,20)
def edge(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.7" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
t(550,43,'批量计算与通信改写：按配置改变图的组织方式',29)
t(550,94,'输入与输出都是 FX 图；三个模块分别选择适用的改写')
a.append('<rect x="35" y="120" width="1030" height="1040" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(175,155,'① 同类算子批处理',['配置 → 规则；按 key 搜索候选，再选互不依赖的子集','如多个 relu → 拼接后一次 relu → select 各结果','未满阈值、无规则或缺依赖：保留对应计算'],'#e3f3ff','#368dc0')
b(395,115,'其他模式 / B2B（不属于本篇三个子模块）',['处在相同 pattern_matcher 开关内；见图重写篇'],'#f4f4f4','#888')
b(585,155,'② TP 计算与通信组合',['开启 _micro_pipeline_tp 才调用','匹配 all-gather / reduce-scatter 与相邻 matmul','检查后组合为 fused 算子；条件不足则保留'],'#e5f6e5','#40a147')
b(815,155,'③ DDP 通信分桶与等待安排',['开启 _fuse_ddp_communication 才调用','按配置顺序执行 pass；分桶/合并与 wait 排序可组合','保留通信完成后才能读取对应结果的依赖'],'#ffe4e4','#ef5555')
b(1040,95,'返回同一张 graph，继续图准备收尾',['本模块不执行多卡通信，也不交出编译后函数'],'#f0e5ff','#9959c6')
edge('M475 150 V175');edge('M475 330 V395');edge('M475 510 V585');edge('M475 740 V815');edge('M475 970 V1040')
edge('M475 150 H1040 V550 H475 V585',True);t(946,369,'pattern_matcher',16);t(946,397,'关闭',18)
edge('M475 550 H1005 V780 H475 V815',True);t(934,660,'TP 关闭',18)
edge('M475 780 H970 V1005 H475 V1040',True);t(914,890,'DDP 关闭',18)
t(550,1200,'实线：对应开关开启后的调用次序；虚线：关闭时的旁路',19)
t(550,1240,'①②③互不要求都有候选；图改写结果仍由后续 Lowering 与运行时消费',19)
a.append('</g></svg>');svg=''.join(a);(P/'batch-communication.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'batch-communication.png'))
im=Image.open(P/'batch-communication.png').convert('RGB');im.thumbnail((770,1000));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'batch-communication-review.png')
