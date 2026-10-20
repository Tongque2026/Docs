"""fallback 的编译期描述与运行时调用；三个职责在构造与生成时重复参与。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="1950" viewBox="0 0 1300 1950"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1300" height="1950" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(y,text,z=21):s.append(f'<text x="650" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(y,h,title,lines,color):
 fill,stroke=color;s.append(f'<rect x="110" y="{y}" width="1080" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(y+36,title,25)
 for i,line in enumerate(lines):t(y+76+i*31,line)
def arrow(a,b,label=''):
 s.append(f'<path d="M650 {a} V{b}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
 if label:s.append(f'<text x="677" y="{(a+b)/2+6}" font-size="18">{escape(label)}</text>')
t(44,'外部调用与 fallback 协议：描述一次调用，保留结构与副作用',29)
s.append('<rect x="35" y="85" width="1230" height="1470" rx="34" fill="#fff9da" stroke="#d7a000" stroke-width="3"/>')
t(125,'编译阶段：接收 target、IR 参数、kwargs 和当前图状态',22)
box(155,160,'① 注册、派发与布局约束',['显式登记 / 未知 target 策略 / 节点级 fallback；不能给任意 Python 函数登记','要求实现输入；布局约束可能复制成连续存储','未知且不允许、decomposition 冲突等条件 → 编译拒绝'],('#e3f3ff','#368dc0'))
box(375,165,'② 参数物化与输出描述：准备调用并重推导属性',['展平 args / kwargs，分离 IR 与非 Tensor 参数，保存 unflatten 规则','realize 输入、冻结布局，按当前 stride 重建通常的 Fake 调用','取得输出结构 / dtype / device / stride / 符号描述，不是模型答案'],('#e5f6e5','#40a147'))
box(600,165,'③ 调用生成与副作用：构造 packed 节点时核对 schema',['functional / view / 受支持 mutable 分支分别处理','登记 alias_names、MutationOutput、side effects 等依赖要求','复制后写入还需共享 mutation 传播写回原输入；不支持的 schema 拒绝'],('#f0e5ff','#9959c6'))
box(825,165,'② 参数物化与输出描述：交回输出 IR',['一个 packed FallbackKernel 可对应多个 MultiOutput','tuple / list / dict 路径取同一次结果；NoneLayout 表示没有普通返回值','不支持的输出、设备推断或符号绑定条件仍可失败'],('#e5f6e5','#40a147'))
box(1050,125,'共享：依赖、调度与存储维护',['外部调用前后可以有普通生成 kernel；按读写 / 别名约束安排顺序','没有 Tensor 返回值的写入仍需保留，不能作为无用计算删除'],('#f2f2f2','#888'))
box(1235,190,'③ 调用生成与副作用：wrapper 生成调用文本',['unflatten 还原参数；Python 发 torch.ops 调用和结果取项','NoneLayout：发调用、不接普通结果；多个取项不重复执行算子','C++ / AOT 另有 C shim、runtime dispatch、序列化及 ABI 要求','完成源码后交给编译加载，等待调用者传入真实输入'],('#f0e5ff','#9959c6'))
for a,b in [(315,375),(540,600),(765,825),(990,1050),(1175,1235)]:arrow(a,b)
t(1495,'三个职责交替参与：① 注册、派发与布局约束　② 参数物化与输出描述　③ 调用生成与副作用',19)
box(1640,165,'运行阶段：dispatcher 调用目标设备已有实现',['前置 kernel 物化输入 → 一次外部调用 → 取项 / 后续 kernel / 必要写回','Fake 描述成功不等于设备已有数值 kernel，也不自动搬到 CPU','本篇 CPU 五组各运行两次；GPU 只验证 Fake 输出与注册表'],('#fff0df','#dc8d35'))
arrow(1555,1640,'加载后由调用者触发')
t(1880,'实线：编译控制 / 数据交接，跨阶段箭头注明触发。灰框为共享职责',20)
s.append('</g></svg>');svg=''.join(s);(P/'external-fallback.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'external-fallback.png'))
im=Image.open(P/'external-fallback.png').convert('RGB');im.thumbnail((850,1300));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'external-fallback-review.png')
