"""按完整设备缓存实例，创建成功才写缓存；方法分别委托。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="1910" viewBox="0 0 1450 1910"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1450" height="1910" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,v,z=22):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(v)}</text>')
def b(y,h,title,lines,c):
 s.append(f'<rect x="70" y="{y}" width="1110" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(625,y+36,title,25)
 for i,v in enumerate(lines):t(625,y+78+i*33,v)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(725,45,'调度实例与后端委托：注册表按类型，实例缓存按完整设备',28)
b(90,115,'输入：torch.device + 当前 Scheduler，已登记 Scheduling 工厂',['调用前可由 Lowering 的 decode_device 补编号；get_backend 不自行归一化'],('#f2f2f2','#888'))
b(260,135,'get_backend：查 self.backends[device]',['cuda:0 / cuda:1 是不同 key；同一 Scheduler 内复用','命中直接返回；未命中进入 create_backend'],('#e1f2ff','#368dc0'))
b(455,180,'create_backend：检查与类型选择',['cuda/xpu 编号缺失 → 断言失败；随后才登记图设备信息','按 device.type 查注册表；缺失 → Unsupported device type','Triton 不可用时检查 CUDA 属性 / GPU 分类；条件不符即失败'],('#e1f2ff','#368dc0'))
b(695,140,'device_scheduling(self)：构造当前图的 Scheduling',['self 是真实 Scheduler；不是能力查询中的 None 探测','组合后端会建立三个子 delegate，保留同一个 Scheduler'],('#e5f6e5','#40a147'))
b(895,140,'构造成功才写入 self.backends[device]',['失败没有缓存实例，但此前图设备信息不会自动回滚','修改全局注册表不替换已经缓存的当前图实例'],('#e5f6e5','#40a147'))
b(1095,125,'返回：当前设备对应的同一 Scheduling 对象',['供分组、融合、生成与 flush 等阶段重复使用'],('#fff4ce','#d6a000'))
b(1290,205,'BaseScheduling / CUDACombinedScheduling：按具体方法处理',['基类：九项未实现接口；空能力 / ready=False / priority=0 有默认值','组合：普通节点、能力、分组、同步、flush 等直接委托 Triton','模板生成：CUDA C++ → ROCm C++ → Triton；前两者拒绝非空 epilogue','融合方法各有自己的分支，不统一经过 choose_node_backend'],('#f0e5ff','#9959c6'))
b(1570,140,'交出结果与生成状态：kernel / wrapper 消费后端方法',['具体循环、模板渲染、合并节点与运行时由其各自模块实现','本层创建编译器对象与组织委托，不在这里执行模型 Tensor 计算'],('#f2f2f2','#888'))
for a,z in [(205,260),(395,455),(635,695),(835,895),(1035,1095),(1220,1290),(1495,1570)]:e(f'M625 {a} V{z}')
e('M1180 325 H1240 V1158 H1180');s.append('<text x="1260" y="715" font-size="18">实例命中：</text><text x="1260" y="747" font-size="18">跳过创建</text>')
t(725,1790,'BaseScheduling 可零参数构造，但不提供接收 Scheduler 的构造器；新设备还需满足实例构造契约。',19)
t(725,1840,'真实 CPU 编译验证实例复用；GPU 属性、设备 worker 与子 delegate 的协议 fixture 明确区分。',19)
s.append('</g></svg>');svg=''.join(s);(P/'backend-dispatch.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'backend-dispatch.png'))
im=Image.open(P/'backend-dispatch.png').convert('RGB');im.thumbnail((950,1250));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'backend-dispatch-review.png')
