"""图收尾叶子：排序状态机与构造节点迁移分析。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
class Draw:
 def __init__(self,title,subtitle,h):
  self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{h}" viewBox="0 0 1200 {h}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1200" height="{h}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">'];self.t(600,43,title,29);self.t(600,94,subtitle,20)
  self.a.append(f'<rect x="30" y="120" width="1140" height="{h-240}" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
 def t(self,x,y,s,z=20):self.a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
 def b(self,x,y,w,h,title,lines,i=0,z=20):
  f,s=[('#e3f3ff','#368dc0'),('#e5f6e5','#40a147'),('#fff3c9','#dca300'),('#ffe4e4','#ef5555'),('#f0e5ff','#9959c6'),('#f4f4f4','#888')][i]
  self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" fill="{f}" stroke="{s}" stroke-width="3"/>');self.t(x+w/2,y+35,title,23)
  for j,line in enumerate(lines):self.t(x+w/2,y+78+j*30,line,z)
 def e(self,p,dash=False):self.a.append(f'<path d="{p}" fill="none" stroke="#222" stroke-width="2.8" marker-end="url(#arr)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
 def save(self,key,note):
  self.t(600,self.h-75,note,19);self.t(600,self.h-30,'实线：本轮处理；虚线：继续扫描；条件和失败出口以文字标明',18);self.a.append('</g></svg>');s=''.join(self.a);(P/f'{key}.svg').write_text(s);cairosvg.svg2png(bytestring=s.encode(),write_to=str(P/f'{key}.png'))
  im=Image.open(P/f'{key}.png').convert('RGB');im.thumbnail((800,1100));gray=ImageOps.grayscale(im).convert('RGB');canvas=Image.new('RGB',(im.width*2,im.height),'white');canvas.paste(im,(0,0));canvas.paste(gray,(im.width,0));canvas.save(P/f'{key}-review.png')
d=Draw('顺序与入口一致性：先修数据顺序，最后刷新 forward','排序只修改节点链表；不改变算子参数、metadata 或数值意图',1540)
d.b(160,155,880,125,'建立三个集合与游标',['pending=逆序节点列表；ready=空；waiting=依赖→使用者','cursor=None；pop 从原图前端的候选开始'],0)
d.b(160,355,880,125,'弹出一个节点，检查 args / kwargs 中的输入',['_args 递归收集 Node，保留重复；常量不形成等待项','waiting_for=尚未进入 ready 的输入节点'],1)
d.b(70,595,470,155,'有未就绪输入：挂起',['等待 waiting_for 最后一项','waiting[该输入].append(当前节点)','暂不移动当前节点'],2)
d.b(660,595,470,155,'全部输入就绪：放到正确位置',['加入 ready，必要时接到 cursor 后','更新 cursor，唤醒等待此节点的用户','唤醒列表反转后加入 pending'],3,z=18)
d.e('M600 280 V355');d.e('M600 480 V535 H305 V595');d.e('M600 535 H895 V595')
d.e('M305 750 V810 H600 V860');d.e('M895 750 V810 H600 V860')
d.b(160,860,880,125,'pending 仍有节点就继续；否则检查结束条件',['waiting 必须为空，ready 必须覆盖整个图','循环引用或外图依赖未解决 → AssertionError'],4)
d.e('M160 920 H45 V320 H600 V355',True);d.t(102,560,'继续',18)
d.b(160,1060,880,115,'交回 post-grad 的跨模块处理',['设备迁移 → 增量属性更新 → 写入 / 通信恢复'],5)
d.b(160,1240,880,125,'最后 recompile、记录图、lint',['recompile 刷新 FX Python forward，不是编译 kernel','lint 检查结构，不证明数值、别名或写入次序正确'],0)
d.e('M600 985 V1060');d.e('M600 1175 V1240');d.save('order-entry','已经合法的顺序保持不变；未编码为数据依赖的副作用先后不由排序器修复')
d=Draw('构造节点设备迁移：从使用链分析到只改 device 参数','本图表示 FX 改写；所有实验使用 FakeTensor，不发生真实 GPU 分配',1470)
d.b(160,155,880,155,'扫描候选和目标设备',['ATen / prims 构造，且 kwargs.device 精确等于 torch.device("cpu")','从 Tensor metadata 收集目标设备；没有候选或目标数量不为 1 → 返回','已有 cuda:0 与 cuda:1 算两个设备'],0,z=19)
d.b(160,370,880,155,'建立 CPU 入度、来源依赖与构造关联组',['cpu_indeg 按参数出现次数计数；每个构造先依赖自己','构造节点放入工作栈；关联组用于共同阻断','constructor_dependencies 保存每条已到达路径的构造来源'],1)
d.b(160,585,880,185,'逐个弹出节点，检查每个使用者',['遇到输出保护、自定义 target、特殊 index_put 等阻断：记录来源并停止该节点','允许 CPU 输入且已输出目标设备：终止这条迁移传播','其余使用者 CPU 入度减 1；到 0 才加入工作栈','合并当前来源和使用者来源，保留关联构造集合'],2,z=18)
d.e('M600 310 V370');d.e('M600 525 V585')
d.b(160,845,880,125,'工作栈处理完：汇总不能迁移的来源',['CPU 入度残留会阻断已关联构造；再扩散到等价关联组','movable=所有候选减去阻断集合'],3,z=19)
d.e('M600 770 V845');d.e('M160 690 H45 V550 H600 V585',True);d.t(95,800,'继续弹栈',17)
d.b(160,1040,880,155,'只对可迁移构造改 kwargs.device',['复制 kwargs，将 device 设为唯一目标设备，再赋回节点','节点身份与原 metadata 不自动改变','调用方随后增量更新属性，再做写入恢复与入口收尾'],4)
d.e('M600 970 V1040');d.save('constructor-devices','allow_outputs 默认 False；只能说明本 pass 是否改写，不能由此推断已验证 GPU 性能')
