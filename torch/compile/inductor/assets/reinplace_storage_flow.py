"""原地化核心：反向建表、正向决策、扫描后统一删除。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
W,H=1300,1420
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1300" height="1420" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(x,y,w,h,title,lines,fill,stroke,z=20):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+35,title,23)
 for j,s in enumerate(lines):t(x+w/2,y+78+j*31,s,z)
def e(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.8" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
t(650,43,'存储可写性与原地化：先检查旧值是否仍会被观察',30)
t(650,94,'输入：已有 metadata 的 FX 图；输出：修改后的 target、clone 决策与重接关系',21)
a.append('<rect x="25" y="120" width="1250" height="1165" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(200,155,900,125,'反向建表：保留原图顺序和存储关系',['node_order、storage_to_nodes；登记输入 / get_attr 的 copy 回写','同一存储的所有 view 一起参与后续旧值使用检查'],'#e3f3ff','#368dc0')
b(200,350,900,125,'正向逐节点：按 target 选择处理分支',['can_inplace 检查存储、同存储 view 的后续使用与回写边界','没有候选的节点直接继续；不是把所有算子改成原地版本'],'#e5f6e5','#40a147')
b(75,600,350,175,'普通候选',['can_inplace 且 extra_check 通过','更换 target 为原地算子','登记可删除的 copy','广义 scatter 使用专门策略'],'#fff3c9','#dca300',17)
b(475,600,350,175,'功能化 / Triton 包装',['逐个参数或 base 检查可写性','同一存储只允许一项复用','记录仍需 clone 的名单','交给后续分解消费'],'#ffe4e4','#ef5555',17)
b(875,600,350,175,'foreach 候选',['每项都有对应 copy 回写','列表存储互不重复且逐项可写','替换 foreach target','登记各项冗余 copy'],'#f0e5ff','#9959c6',17)
b(200,900,900,125,'当前节点结束：保留决策，继续正向扫描',['target / metadata / kwargs 可立即更新；待删除节点先放 replace_dict','此处不执行真实 Tensor 运算；包装节点仍需后续分解'],'#e5f6e5','#40a147')
b(200,1120,900,125,'全部扫描完成：追到替换链终点再重接、删除',['逐项 replace_all_uses_with，erase_node；正常返回 None','普通原地化与包装 clone 决策在同一轮共享存储判断'],'#f0e5ff','#9959c6')
e('M650 280 V350');e('M650 475 V530 H250 V600');e('M650 530 V600');e('M650 530 H1050 V600')
e('M250 775 V830 H400 V900');e('M650 775 V900');e('M1050 775 V830 H900 V900')
e('M200 965 H50 V315 H650 V350',True);t(105,850,'继续',18);t(105,880,'下一节点',18)
e('M650 1025 V1120');t(650,1080,'扫描结束才执行下面的删除',19)
t(650,1330,'实线：本轮改图；虚线：扫描循环。后续分解把 clone 决策变成实际操作',20)
t(650,1380,'普通 Tensor 的值相同，不足以证明输入修改、旧值和存储别名语义都正确',20)
a.append('</g></svg>');svg=''.join(a);(P/'reinplace-storage.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'reinplace-storage.png'))
im=Image.open(P/'reinplace-storage.png').convert('RGB');im.thumbnail((850,1000));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'reinplace-storage-review.png')
