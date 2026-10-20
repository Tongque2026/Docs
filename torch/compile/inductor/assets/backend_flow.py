"""设备后端接口：注册、后端实例与循环生成，标出共享 wrapper。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent;H=1640
a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="{H}" viewBox="0 0 1300 {H}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1300" height="{H}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=20):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(y,h,title,lines,i,z=20):
 fill,stroke=[('#e3f3ff','#368dc0'),('#e5f6e5','#40a147'),('#fff3c9','#dca300'),('#f0e5ff','#9959c6'),('#f4f4f4','#888')][i]
 a.append(f'<rect x="100" y="{y}" width="1100" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(650,y+35,title,24)
 for j,s in enumerate(lines):t(650,y+78+j*31,s,z)
def e(d):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.8" marker-end="url(#arr)"/>')
t(650,43,'后端与循环组织：设备类型怎样选到代码生成实现',30)
t(650,94,'三个职责：① 后端注册与能力查询　② 调度实例与后端委托　③ 循环展开与 kernel 分组',20)
a.append('<rect x="35" y="120" width="1230" height="1360" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(155,155,'① 后端注册与能力查询',['设备类型 → Scheduling、Python wrapper、C++ wrapper 三项关联','DeviceOpOverrides 独立登记设备操作；不是重命名设备就自动生成','能力查询会创建 Scheduling(None)，它不是某张图的运行后端'],0)
b(375,125,'跨模块：GraphLowering 先建立 wrapper',['按当前图设备类型查询①，选择 wrapper 类与设备操作对象','后续 kernel 定义和调用写入同一个 wrapper；详见 wrapper 子篇'],4)
b(565,190,'② 调度实例与后端委托',['Scheduler 按完整 device 缓存实例，再从①取 Scheduling 类并构造','能力、融合、group_fn 与 codegen_node/template 由该实例提供','一个设备类型可委托多个实现：CUDACombinedScheduling 选择模板或普通路径','找不到注册类会报错；当前 GPU / Triton 条件另受已有分类函数约束'],1,z=19)
b(820,155,'③ 循环展开与 kernel 分组：准备生成环境',['C++：节点范围 → CppKernelProxy、循环 / 向量化 / 尾部','Triton：迭代 / 归约范围 → 节点计划、分块与 mask 环境','输入是调度节点及其 IR body；不是一份待执行的真实 Tensor'],2)
b(1040,125,'跨模块：元素表达式与目标代码',['在当前 kernel 环境解释 load / add / store 等元素表达式','返回生成区文本、临时值与参数使用；完整实现归兄弟模块'],4)
b(1230,155,'③ 汇总定义与调用，交回已建立的 wrapper',['C++ KernelGroup 可积累循环，flush 时生成函数；空组不发调用','后端定义 kernel，并登记参数、编译请求和调用语句','后续 wrapper.generate、编译加载才形成可运行的图入口'],3)
for y,z in [(310,375),(500,565),(755,820),(975,1040),(1165,1230)]:e(f'M650 {y} V{z}')
t(650,1435,'最终交付：kernel 源码 + 参数 / 调用要求；运行时才用真实输入产生结果',21)
t(650,1530,'实线：一张图的代码生成控制主线；①被后续步骤按需查询，不会每次重新注册',19)
t(650,1580,'灰框为共享模块。注册与接口实验不等于新硬件 kernel 已经生成或运行',19)
a.append('</g></svg>');svg=''.join(a);(P/'backend-loops.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'backend-loops.png'))
im=Image.open(P/'backend-loops.png').convert('RGB');im.thumbnail((850,1100));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/'backend-loops-review.png')
