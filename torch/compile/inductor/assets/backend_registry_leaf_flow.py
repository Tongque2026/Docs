"""注册叶子中的两张表与能力探测；框是动作，不再拆模块。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="1760" viewBox="0 0 1450 1760"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1450" height="1760" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,v,z=21):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(v)}</text>')
def b(x,y,w,h,title,lines,c):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{c[0]}" stroke="{c[1]}" stroke-width="3"/>');t(x+w/2,y+36,title,24)
 for i,v in enumerate(lines):t(x+w/2,y+80+i*32,v)
def e(d):s.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
t(725,45,'后端注册与能力查询：两张表，分别交出生成实现与设备代码接口',28)
b(160,90,1130,120,'输入：设备类型名 + Scheduling / wrapper 实现；或设备操作对象',['登记、默认初始化、精确查表和能力探测，触发条件不同'],('#f2f2f2','#888'))
e('M725 210 V245 H360 V290');e('M725 245 H1090 V290')
b(50,290,620,220,'后端实现登记与默认初始化',['显式登记直接覆盖同 key','默认 CPU / CUDA / XPU 仅在 Scheduling 为 None 时补入','重命名 PrivateUse1：发现三个实现，全部为真才登记','初始化有 lru_cache；不是每次 getter 都自动初始化'],('#e1f2ff','#368dc0'))
b(780,290,620,220,'register_device_op_overrides / 默认模块导入',['显式登记直接保存对象、覆盖同 key','lookup 只在整张表为空时导入默认 CUDA / XPU 模块','模块导入缓存与对象字典是两套独立状态','不执行 set_device / stream / 同步'],('#e5f6e5','#40a147'))
e('M360 510 V565');e('M1090 510 V565')
b(50,565,620,155,'device_codegens：类型 key → DeviceCodegen',['scheduling / wrapper_codegen / cpp_wrapper_codegen','省略 C++ 槽位默认 type(None)，不是可用实现'],('#e1f2ff','#368dc0'))
b(780,565,620,155,'device_op_overrides_dict：类型 key → 对象',['保存实际对象引用','它与左侧 Scheduling/wrapper 登记没有自动联动'],('#e5f6e5','#40a147'))
e('M360 720 V780');e('M1090 720 V780')
b(50,780,620,165,'get_scheduling / get_wrapper_codegen',['精确 key 查表；缺失返回 None','wrapper 根据 cpp_wrapper 选择槽位','单独调用这些 getter 不触发默认初始化'],('#e1f2ff','#368dc0'))
b(780,780,620,165,'get_device_op_overrides',['已有 key 返回同一对象；缺失隐式返回 None','表非空时不会因为缺少某个 key 而导入默认模块','清空表也不会自动重执行已导入模块的顶层登记'],('#e5f6e5','#40a147'))
e('M360 945 V1010');t(205,980,'能力查询路径',18);e('M1090 945 V1010');e('M670 860 H725 V1260 H360');s.append('<rect x="680" y="955" width="90" height="57" fill="white"/>');t(725,978,'wrapper 类',16);t(725,1000,'直接交出',16)
b(50,1010,620,225,'get_backend_features / has_backend_feature',['先初始化，再取得 Scheduling','torch.device 对象用 .type 查表；字符串保留原 key','Scheduling(None) → get_backend_features(device)','每次能力查询可新建探测实例，不是读取缓存的能力值','has_backend_feature 校验 Enum 后做成员查询'],('#f0e5ff','#9959c6'))
b(780,1010,620,225,'交给 wrapper 的代码生成接口',['实际对象生成 import / set_device / guard 等字符串','基类四个方法均抛 NotImplementedError','具体字符串与运行调用归 wrapper 设备入口','登记成功不代表设备 Tensor、kernel 或 runtime 已实现'],('#fff4ce','#d6a000'))
e('M360 1235 V1290 H725 V1350');e('M1090 1235 V1290 H725')
b(160,1350,1130,140,'输出：生成类/工厂、能力集合、设备代码接口对象',['后续 GraphLowering 与 Scheduler 消费这些对象','本层本身没有分配新设备 Tensor，也不编译/执行 kernel'],('#f2f2f2','#888'))
t(725,1600,'默认 Scheduling 工厂在调用时读取后端配置；初始化缓存不等于冻结所有后续查询结果。',20)
t(725,1650,'源码、独立进程协议与先前真实 CPU 编译消费证据分别记录；没有新设备硬件运行。',19)
s.append('</g></svg>');svg=''.join(s);(P/'backend-registry-leaf.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'backend-registry-leaf.png'))
im=Image.open(P/'backend-registry-leaf.png').convert('RGB');im.thumbnail((1000,1250));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'backend-registry-leaf-review.png')
