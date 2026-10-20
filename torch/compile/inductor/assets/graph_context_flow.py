"""图状态、动态作用域与设备记录；共享职责与调用后恢复分开标注。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="1530" viewBox="0 0 1250 1530"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1250" height="1530" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,text,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(y,h,title,lines,color):
 fill,stroke=color;s.append(f'<rect x="90" y="{y}" width="1000" height="{h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(590,y+36,title,25)
 for i,line in enumerate(lines):t(590,y+76+i*31,line)
def arr(y,z):s.append(f'<path d="M590 {y} V{z}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(625,45,'图上下文与符号状态：同一张图持续积累编译结果',29)
box(90,125,'输入：gm + 示例输入 + ShapeEnv（可选）+ 编译模式',['收到已有环境就复用；没有就创建本图环境','调用者先安装 V.fake_mode；示例提供设备、尺寸与步长'],('#f3f3f3','#888'))
s.append('<rect x="35" y="255" width="1110" height="940" rx="32" fill="#fff9d9" stroke="#d6a000" stroke-width="3"/>')
box(285,165,'建立 GraphLowering 状态',['冻结新增 deferred assert；浅复制等待表，建立 SizeVarAllocator','初始化输入 / 常量 / buffer / operation / 名称映射','const_module 共享部分容器；make_subgraph 另建状态、复用 ShapeEnv'],('#e3f3ff','#368dc0'))
box(510,135,'构造完成后，调用者安装 V.graph 作用域',['Virtualized 通过 threading.local 找到当前编译对象','保存 prior 并立即安装新值；with 退出时 finally 恢复'],('#e5f6e5','#40a147'))
box(705,190,'随节点处理更新本图状态（重复）',['共享：输入登记 / 节点派发建立 IR，并安装当前节点','尺寸：静态常数，或沿用 / 创建符号尺寸与步长','设备：type 集合 + index 集合；完整 device → 首次来源节点','能力：按设备类型查询 BackendFeature，缓存归当前图'],('#fff0cf','#d8a000'))
box(955,165,'交给后续模块：保留 GraphLowering 与已建立的 IR',['共享：布局选择、Scheduler、wrapper、编译加载读取同一状态','收集到多个设备类型，不代表 wrapper 允许它们混用','仅登记设备名，不会实现 allocator、算子、guard 或 launcher'],('#f3f3f3','#888'))
for y,z in [(215,285),(450,510),(645,705),(895,955)]:arr(y,z)
t(590,1160,'本篇为职责叶子；图中动作不另拆模块。灰框 / 共享项由已有相邻模块负责',18)
box(1260,145,'退出本次编译作用域：恢复先前 handler',['正常退出、嵌套返回、异常退出都恢复 prior','图对象和 IR 仍由调用者持有；V 不再冒充永久的进程级当前图'],('#f0e5ff','#9959c6'))
arr(1195,1260)
t(625,1470,'实线：编译控制与状态交接。本篇只研究编译状态，未生成或执行设备 kernel',19)
s.append('</g></svg>');svg=''.join(s);(P/'graph-context.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'graph-context.png'))
im=Image.open(P/'graph-context.png').convert('RGB');im.thumbnail((850,1100));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'graph-context-review.png')
