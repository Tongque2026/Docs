"""候选/编译测试/计时缓存与交付；明确快路、缓存绕过与延迟消费时点。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1960" viewBox="0 0 1400 1960"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1400" height="1960" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,text,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(y,h,title,lines,color):
 fill,stroke=color;s.append(f'<rect x="90" y="{y}" width="1120" height="{h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(650,y+36,title,25)
 for i,line in enumerate(lines):t(650,y+76+i*31,line)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
def arrow(a,b,label=''):
 e(f'M650 {a} V{b}')
 if label:s.append(f'<text x="676" y="{(a+b)/2+6}" font-size="18">{escape(label)}</text>')
t(700,44,'模板候选与方案选择：生成描述、按需比较、交回实现 IR',29)
box(90,110,'输入：算子名、输入 IR、输出 layout 与候选条件',['共享的专用 lowering 决定加入哪些实现；本层不重新介绍上游编译流程'],('#f2f2f2','#888'))
box(250,140,'① 候选对象与模板生成',['Caller 保留实现 hash、输入/layout、编译或可调用描述、output_node','模板还携带渲染与 BenchmarkRequest；不是只有一个候选名字'],('#e3f3ff','#368dc0'))
box(440,145,'③ 计时缓存与结果交付：先过滤与分流',['过滤 None；空集 → 拒绝；单个非 CUDA 模板 → output_node 后结束','多候选或 CUDATemplateCaller 例外，才继续下方路径'],('#f0e5ff','#9959c6'))
box(645,155,'② 预编译、校验与计时：准备编译任务',['可先查询③是否已有完整计时，命中可省略预编译','按条件提交 Future，保留等待函数；这还不是模型执行','延迟选择对象返回前，也可能已提交预编译'],('#e5f6e5','#40a147'))
box(860,165,'③ 计时缓存与结果交付：确定消费时点',['立即路径现在需要计时；延迟路径先返回 MultiTemplateBuffer','延迟对象首次读取 choice_timings 时，才触发下面的查表/测时','CPU 或提供 input_gen_fns 时禁用该延迟路径'],('#f0e5ff','#9959c6'))
box(1085,130,'③ 先调用②等待函数，再查询候选计时',['键含实现 hash、输入布局、精度与系统描述','命中 → 右侧交付；缺失且允许测试 → 交给②'],('#f0e5ff','#9959c6'))
box(1275,160,'② 预编译、校验与计时：实际执行候选测试',['生成实际输入，正确解释布局、基地址与 offset','VERIFY 开启时做相应校验；实际计时可产生失败/inf','把测得映射交回③更新缓存；不是人工 score 充当耗时'],('#e5f6e5','#40a147'))
box(1495,155,'③ 选择并交付 output_node，或保存延迟结果',['正常完整计时取最小；空表/首候选缺失有退路；全非有限结果拒绝','得到外部调用 / 模板 buffer 等 IR，延迟对象可供调度继续确定','计时缓存复用不固定模型答案，也不缓存返回 Tensor'],('#f0e5ff','#9959c6'))
for a,b,label in [(200,250,''),(390,440,''),(585,645,'其余候选'),(800,860,''),(1025,1085,'需要结果时'),(1215,1275,'缺失且允许测试'),(1435,1495,'实际结果')]:arrow(a,b,label)
e('M1210 1150 H1260 V1570 H1210');s.append('<text x="1280" y="1330" font-size="18">命中：</text><text x="1280" y="1360" font-size="18">跳过测时</text>')
box(1720,120,'共享：调度 / 设备 codegen / 编译加载',['接收选择后的 IR，再形成正式可执行图入口；GPU 等路径仍需对应设备实现'],('#f2f2f2','#888'))
arrow(1650,1720)
t(700,1900,'实线：控制和结果交接；延迟时点在框中注明。①②③是三个职责，不是固定只执行一次的阶段',19)
s.append('</g></svg>');svg=''.join(s);(P/'algorithm-selection.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'algorithm-selection.png'))
im=Image.open(P/'algorithm-selection.png').convert('RGB');im.thumbnail((950,1400));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'algorithm-selection-review.png')
