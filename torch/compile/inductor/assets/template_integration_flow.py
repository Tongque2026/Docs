"""专用 lowering 三类接入，选择器共享；复合路径不伪装成必经的 autotune。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1770" viewBox="0 0 1400 1770"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1400" height="1770" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,text,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(x,y,w,h,title,lines,color,z=21):
 fill,stroke=color;s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+36,title,23)
 for i,line in enumerate(lines):t(x+w/2,y+77+i*32,line,z)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(700,45,'模板与复合算子接入：按算子、形状与设备条件形成不同 IR',29)
box(130,90,1140,155,'输入：target + 输入 IR + shape / stride / dtype / device + 编译条件',['专用 lowering 按当前算子和条件分流，下面三项不是必经的三个阶段','本层交回输出 IR 和分组 / 副作用状态；不直接交回模型数值'],('#f2f2f2','#888'))
s.append('<rect x="25" y="300" width="1350" height="1070" rx="34" fill="#fff9da" stroke="#d7a000" stroke-width="3"/>')
e('M700 245 V345 H250 V400');e('M700 345 V400');e('M700 345 H1150 V400')
box(50,400,400,295,'① 矩阵与卷积的候选接入',['物化 / 整理形状 / 约束布局','按设备、dtype、架构、库等加候选','候选携带输入、layout、模板参数','无候选：按算子规则回退或拒绝','某些 CPU bmm 改走②'],('#e3f3ff','#368dc0'),19)
box(500,400,400,295,'② 列表与复合计算',['复用 Pointwise / Reduction 等 IR','foreach 按位置和设备整理计算','满足能力与条件才登记 operation list','返回普通输出及列表 / 写入关系','不等于已生成 native foreach'],('#e5f6e5','#40a147'),19)
box(950,400,400,295,'③ 用户 kernel 与专用调用',['用户描述 / grid / 常量 / Tensor','参数顺序、constexpr 与写入分开','建立专用 IR 和 MutationOutput','effect / collective 还带顺序约束','真实执行仍需设备实现'],('#f0e5ff','#9959c6'),19)
e('M450 590 H500');t(475,573,'改写',16)
box(50,795,400,220,'共享：算法选择器',['收到 choices、输入和输出 layout','外部节点 / 模板 / 可选延迟结果','本篇 CPP 只有一个候选','直接 output_node，未 benchmark'],('#f2f2f2','#888'),19)
e('M250 695 V795');e('M250 1015 V1160');e('M700 695 V1160');e('M1150 695 V1160')
s.append('<text x="728" y="886" font-size="19">无需模板选择</text>');s.append('<text x="1177" y="886" font-size="19">尚未启动 kernel</text>')
box(100,1160,1200,155,'交回共享 GraphLowering 状态，供后续调度使用',['ExternKernelOut / CppTemplateBuffer 等结果，或普通 ComputedBuffer、专用调用节点','输入布局、常量权重、operation list、mutation / effect 关系按本次路径保留','当前 CPU 模板权重被预打包，但逻辑输出 shape 仍为 [8,16]'],('#f2f2f2','#888'))
box(130,1440,1140,160,'共享：设备调度 / 代码生成 → wrapper → 编译加载',['接收本层 IR，再生成真实目标 kernel 或外部调用；运行时才产生模型结果','登记能力、候选或 grid 不自动补齐新设备的编译器、launcher 和算子','本篇七个 CPU 程序实际编译运行；GPU gate / 用户 kernel 仅协议证据'],('#fff0df','#dc8d35'))
e('M700 1370 V1440')
t(700,1675,'实线：编译期数据交接；横向改写箭头表示形状分支可转入普通复合计算',20)
t(700,1730,'灰框为共享设施。真实 CPP 模板成功不代表测得加速；本篇成功 CPU 路径均未构造 benchmark',20)
s.append('</g></svg>');svg=''.join(s);(P/'template-integration.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'template-integration.png'))
im=Image.open(P/'template-integration.png').convert('RGB');im.thumbnail((950,1220));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'template-integration-review.png')
