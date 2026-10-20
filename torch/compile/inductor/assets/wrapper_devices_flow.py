"""入口装配、设备上下文和启动协议；编译生成与运行时明确分组。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1570" viewBox="0 0 1200 1570"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1200" height="1570" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(y,s,size=21):parts.append(f'<text x="600" y="{y}" font-size="{size}" text-anchor="middle">{escape(s)}</text>')
def box(y,h,title,lines,color):
 fill,stroke=color
 parts.append(f'<rect x="115" y="{y}" width="970" height="{h}" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(y+36,title,25)
 for i,line in enumerate(lines):t(y+76+i*31,line)
def arrow(a,b,label=''):
 parts.append(f'<path d="M600 {a} V{b}" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
 if label:parts.append(f'<text x="625" y="{(a+b)/2+5}" font-size="18">{escape(label)}</text>')
t(48,'入口与设备调用：先生成程序，再执行入口',30)
parts.append('<rect x="35" y="85" width="1130" height="1095" rx="30" fill="#fff9db" stroke="#d9a300" stroke-width="3"/>')
t(125,'编译阶段：接收有序调度结果、kernel 名称、参数和设备',23)
box(155,135,'① 入口参数与程序装配：准备程序片段',['选定 wrapper 类与 DeviceOpOverrides；建立缓冲区','写 imports / header / call(args) 前缀；尚未执行真实输入'],('#e3f3ff','#368dc0'))
box(350,135,'② 设备上下文与 stream：登记上下文',['设备切换 → 追加延迟 guard 行；请求 stream 变量','按 device + graph 复用获取语句；import 单独去重'],('#e5f6e5','#40a147'))
box(545,135,'③ kernel 调用与启动参数：登记调用',['CPU 普通调用 / Triton .run / CUDA C++ + stream 指针','可选：另写编译期 autotune 样例片段；不替代正常调用'],('#ffe6e6','#e55b61'))
box(740,125,'共享：存储计划与输出处理',['先规划分配、复用与释放，再解析调用体中的行对象','服务整张图；实现归 wrapper 的两个兄弟模块'],('#f1f1f1','#888'))
box(925,155,'① 入口参数与程序装配：汇总并返回源码',['合并前缀、调用体、输出返回与后缀，产生源码和行映射','交给编译加载：普通 Python 模块加载时执行 async_compile.wait','定义 call(args)；函数体等待真实调用'],('#e3f3ff','#368dc0'))
for a,b in [(290,350),(485,545),(680,740),(865,925)]:arrow(a,b)
t(1135,'②③由后端按调度顺序重复参与；图示是一段设备调用的主线',19)
box(1240,195,'运行阶段：加载后的 call(args) 收到真实 Tensor',['解包输入、args.clear()、绑定符号并检查布局','进入设备 guard → 每次调用获取当前 stream → 启动 kernel','退出上下文并返回输出；clear 列表不会清除 Tensor 数据','GPU 路径本篇验证生成文本与替身协议；CPU 完整编译执行'],('#f0e5ff','#9959c6'))
arrow(1180,1240,'加载后，由调用者触发')
t(1500,'实线表示程序生成主线或标注的运行触发；灰框为共享职责',20)
parts.append('</g></svg>');svg=''.join(parts);(P/'wrapper-devices.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'wrapper-devices.png'))
im=Image.open(P/'wrapper-devices.png').convert('RGB');im.thumbnail((800,1100));canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));canvas.save(P/'wrapper-devices-review.png')
