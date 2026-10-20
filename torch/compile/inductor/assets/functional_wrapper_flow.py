"""功能化包装：名单来源、flatten/unflatten 与必须清空的出口。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent;H=1360
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="{H}" viewBox="0 0 1300 {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1300" height="{H}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(x,y,w,h,title,lines,fill,stroke,z=20):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+35,title,23)
 for j,s in enumerate(lines):t(x+w/2,y+78+j*30,s,z)
def e(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.8" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
t(650,43,'功能化包装分解：把保留 clone 的决策变成实际图操作',29)
t(650,94,'前提：原地化核心已经给出完整 clone 决策；缺省空名单不等于未知时全部 clone',20)
a.append('<rect x="30" y="120" width="1240" height="1120" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(200,155,900,125,'建立本次 PatternMatcherPass，登记三类 Graph handler',['CallFunctionVarArgs 捕获各自完整位置参数与关键字参数','登记完毕再 apply；此刻尚未运行下面的 decomp 闭包'],'#e3f3ff','#368dc0')
b(55,410,370,190,'auto_functionalized v1',['从 node.meta 取保留参数名','转换为 tuple，交 dense 实现','只 clone 名单中的 mutable 参数','输出：原返回项 + mutable 参数结果'],'#e5f6e5','#40a147',17)
b(465,410,370,190,'auto_functionalized v2',['同字段保存的是 base 整数索引','交给 v2 dense，先选 base clone','再按原视图信息重建 mutable 参数','输出：原返回项 + 完整 base 结果'],'#fff3c9','#dca300',17)
b(875,410,370,190,'Triton functional wrapper',['名单来自 kwargs.tensors_to_clone','交 functional dense 实现','字典结果外面再包装一层 tuple','替代图可包含 mutation wrapper'],'#ffe4e4','#ef5555',17)
b(200,710,900,155,'各 handler：flatten 参数 → 建立 decomp → replace_by_example',['spec 保存 args/kwargs 结构；decomp 用 unflatten 恢复，再调用对应 dense','按当前 FakeTensor 输入追踪替代图，复制节点并重接原 getitem 等使用者','run_functional_passes=False：跳过替代图中的 noop / DCE 功能式假设'],'#f0e5ff','#9959c6')
b(200,970,900,125,'apply 返回后：逐类检查原 functional target 是否仍存在',['任何一类残留 → 对应 AssertionError','成功：三类目标都清空，正常返回 None；继续 post-grad 收尾'],'#e5f6e5','#40a147')
e('M650 280 V340 H240 V410');e('M650 340 V410');e('M650 340 H1060 V410')
e('M240 600 V645 H410 V710');e('M650 600 V710');e('M1060 600 V645 H890 V710');e('M650 865 V970')
t(650,1170,'输出仍是 FX 图：clone、布局恢复和底层 mutable 操作已经显式出现',21)
t(650,1290,'实线：匹配命中后的编译期图转换；多个 wrapper 按 matcher 的候选扫描逐个处理',19)
t(650,1330,'三类 functional target 被移除，不代表图中所有高阶 / 自定义算子都必须消失',19)
a.append('</g></svg>');svg=''.join(a);(P/'functional-wrappers.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'functional-wrappers.png'))
im=Image.open(P/'functional-wrappers.png').convert('RGB');im.thumbnail((850,1000));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'functional-wrappers-review.png')
