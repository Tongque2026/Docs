"""通用/设备/环与收益三职责；can_fuse 边界与真正合组分离。"""
from pathlib import Path
from html import escape
import cairosvg
from PIL import Image,ImageOps
P=Path(__file__).resolve().parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="1770" viewBox="0 0 1300 1770"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1300" height="1770" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(y,text,z=21):s.append(f'<text x="650" y="{y}" text-anchor="middle" font-size="{z}">{escape(text)}</text>')
def box(y,h,title,lines,color):
 fill,stroke=color;s.append(f'<rect x="110" y="{y}" width="1080" height="{h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(y+36,title,25)
 for i,line in enumerate(lines):t(y+77+i*31,line)
def arrow(a,b,label='通过'):
 s.append(f'<path d="M650 {a} V{b}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#a)"/>')
 if label:s.append(f'<text x="676" y="{(a+b)/2+6}" font-size="18">{escape(label)}</text>')
t(45,'融合合法性与后端判断：允许生成，不等于已经合组',30)
box(90,125,'共享：候选选择 / 当前组映射',['两个当前调度组：设备、读写、祖先、循环范围与节点顺序','搜索时先过滤；消费候选时映射到当前组，再次判断'],('#f2f2f2','#888'))
s.append('<rect x="45" y="265" width="1210" height="590" rx="32" fill="#fff9da" stroke="#d7a000" stroke-width="3"/>');s.append('<text x="80" y="300" font-size="22">Scheduler.can_fuse 的范围</text>')
box(325,195,'① 通用约束与读写匹配',['类型 / 模板 / 禁融名称 / 完整设备 / 共享数据 / 规模','纵向：读写匹配并排除中间节点；横向：检查距离策略','配置允许时，共享数据检查可能重排循环，修改节点','拒绝 → 返回 False；本候选不进入后端或不再继续'],('#e3f3ff','#368dc0'))
box(590,165,'② 设备融合判断',['纵向 hook 或横向 hook：CPU 范围、SIMD 迭代 / 归约 / tiling','新设备必须给出与自身 codegen 一致的判断','返回 False → 停止本候选；缺失基类 hook → NotImplementedError'],('#e5f6e5','#40a147'))
t(810,'can_fuse=True：只完成这两个职责，尚未调用 backend.fuse',22)
arrow(215,325,'当前组');arrow(520,590)
box(925,165,'③ 环检测与融合收益：先检查新增依赖环',['will_fusion_create_cycle 沿原 operation → 当前融合组映射查路径','已有合组可能带来原 ancestors 中没有的路径','发现环 → 拒绝，保留已有合法分组'],('#f0e5ff','#9959c6'))
arrow(855,925,'can_fuse=True')
box(1160,165,'③ 环检测与融合收益：再判断是否值得合并',['speedup_by_fusion：普通关闭测时路径允许；另有模板分支','CPU 等路径即使配置开启也可跳过；本篇没有 GPU 测时','收益不满足 → 拒绝；通过才交给合组模块'],('#f0e5ff','#9959c6'))
arrow(1090,1160,'没有环')
box(1395,155,'共享：合并与分组维护',['调用 backend.fuse，替换节点，更新原操作到新组映射','重新排序与清理依赖，供下一候选 / 下一轮继续使用','本篇 CPU 对照：允许为 1 组，显式后端否决保留 2 组'],('#f2f2f2','#888'))
arrow(1325,1395)
t(1620,'任一拒绝只结束当前候选；继续处理其他候选。异常不是普通 False',21)
t(1680,'实线：成功主线；框内标明失败出口。合组后的对象仍是编译状态，不是运行结果',20)
t(1730,'协议 / SIMD 案例使用明确元信息 fixture；四个 CPU 案例实际完整编译执行',20)
s.append('</g></svg>');svg=''.join(s);(P/'fusion-legality.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'fusion-legality.png'))
im=Image.open(P/'fusion-legality.png').convert('RGB');im.thumbnail((850,1160));out=Image.new('RGB',(im.width*2,im.height),'white');out.paste(im,(0,0));out.paste(ImageOps.grayscale(im).convert('RGB'),(im.width,0));out.save(P/'fusion-legality-review.png')
