"""模式协议：登记、调用匹配、条件应用与延迟 lowering。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
W,H=1250,1570
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1250" height="1570" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,size=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{escape(s)}</text>')
def b(x,y,w,h,title,lines,i):
    fill,stroke=[('#e3f3ff','#368dc0'),('#e5f6e5','#40a147'),('#fff3c9','#dca300'),('#ffe4e4','#ef5555'),('#f0e5ff','#9959c6')][i]
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+33,title,22)
    for j,s in enumerate(lines):t(x+w/2,y+70+j*29,s,18)
def edge(path,dash=False):a.append(f'<path d="{path}" fill="none" stroke="#222" stroke-width="2.7" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
t(625,42,'模式匹配与子图替换：先收集证据，再应用规则',29)
t(625,93,'三项职责：① 规则构造与登记　② 模式与匹配状态　③ 候选应用与图维护',20)
a.append('<rect x="35" y="125" width="1180" height="1175" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(120,155,720,125,'① 规则构造与登记',['手写 PatternExpr 或追踪 search_fn；建立 entry','patterns[(op,target)] 保存有顺序的规则'],0)
b(120,340,720,125,'③ 候选应用与图维护：建立本轮候选快照',['当前 FX 图 + patterns → 逆序访问根节点','每个根按登记顺序尝试 entry'],2)
b(120,525,720,125,'② 模式与匹配状态：调用 pattern.match(root)',['递归核对算子、参数、常量、共享节点和用户数','返回 Match（捕获与节点）或 FailedMatch'],1)
b(120,710,720,140,'③ 候选应用与图维护：检查条件',['Match 跨 mutation region → 拒绝；extra_check=False → 拒绝','Replacement entry 还可按当前属性重追踪，再生成替代图','通过后调用 entry.apply；不代表已经执行数值计算'],2)
for start,end in [(280,340),(465,525),(650,710)]:edge(f'M480 {start} V{end}')
t(480,907,'entry 类型决定如何改写（都属于③）',21)
b(75,960,340,150,'GraphPatternEntry',['调用 handler(match, FX Node)','handler 负责图修改和删除','可在内部用 replace_by_example'],3)
b(455,960,340,150,'ReplacementPatternEntry',['复制已追踪 replacement_graph','重接输出、传播相关 metadata','删除不再使用的旧节点'],3)
b(835,960,340,150,'LoweringPatternEntry',['插入绑定 Match 的 handler','复制根 metadata、重接和删除','此时 handler 尚未构造 IR'],4)
edge('M480 850 V925 H245 V960');edge('M480 925 H625 V960');edge('M625 925 H1005 V960')
b(910,340,270,510,'③ 扫描控制',['失败，或 A：处理结束','继续本轮已有候选','新建根不自动追加','同一旧根未删除时','仍可能尝试后续 entry','','不自动迭代到不再变化'],2)
edge('M840 587 H910');edge('M840 785 H910')
edge('M245 1110 V1160 H625 V1190');edge('M1005 1110 V1160 H625 V1190');edge('M625 1110 V1190')
t(625,1225,'A：entry 处理结束 → 返回右侧扫描控制（配对连接点）',19)
edge('M1045 340 V312 H480 V340',True)
t(625,1275,'扫描结束：返回应用计数；图已原地修改，调用方负责后续收尾',21)
edge('M1175 1070 H1230 V1320 H1005 V1340',True)
b(785,1340,425,125,'稍后：GraphLowering.call_function',['FX 参数已解释为 TensorBox 等 IR','才调用 handler；运行程序不再调用它'],4)
t(375,1390,'虚线：继续候选，或稍后触发',19)
t(375,1430,'实线：登记、匹配、条件检查与图修改',19)
t(625,1525,'编译图与执行图分开：lowering handler 的参数契约是 IR，不能当普通 Tensor 算子调用',20)
a.append('</g></svg>');svg=''.join(a);(P/'pattern-protocol.svg').write_text(svg)
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'pattern-protocol.png'))
im=Image.open(P/'pattern-protocol.png').convert('RGB');im.thumbnail((800,1100));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'pattern-protocol-review.png')
