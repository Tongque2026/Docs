"""叶子控制流：主路径、跳过路径与独立返回通道。"""
from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent
COLORS=[('#e3f3ff','#368dc0'),('#e5f6e5','#40a147'),('#fff3c9','#dca300'),('#ffe4e4','#ef5555'),('#f0e5ff','#9959c6')]
class Drawing:
    def __init__(self,title,subtitle,height=1110):
        self.h=height;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="{height}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
        self.text(550,42,title,29);self.text(550,95,subtitle,21)
        self.a.append(f'<rect x="35" y="130" width="1040" height="{height-250}" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
    def text(self,x,y,s,size=20):self.a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{escape(s)}</text>')
    def box(self,y,title,lines,i=0,height=110):
        fill,stroke=COLORS[i%5];self.a.append(f'<rect x="130" y="{y}" width="670" height="{height}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');self.text(465,y+34,title,23)
        for j,s in enumerate(lines):self.text(465,y+70+j*29,s,19)
    def arrow(self,d,dash=False):self.a.append(f'<path d="{d}" fill="none" stroke="#333" stroke-width="2.6" marker-end="url(#arr)"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
    def control(self,top,bottom,lines,return_y,exit_note):
        self.a.append(f'<rect x="850" y="{top}" width="200" height="{bottom-top}" rx="22" fill="#fafafa" stroke="#777" stroke-width="2.5"/>')
        self.text(950,top+38,'扫描控制',21)
        for j,s in enumerate(lines):self.text(950,top+85+j*34,s,17)
        self.text(950,bottom-65,exit_note,16)
        self.arrow(f'M950 {top} V{top-25} H825 V{return_y} H465',True)
    def save(self,key,note):
        self.text(550,self.h-83,note,19);self.text(550,self.h-42,'实线：本轮处理；虚线：继续扫描。扫描结束后退出，跳过条件写在框内',17)
        self.a.append('</g></svg>');svg=''.join(self.a);(P/f'{key}.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{key}.png'))
d=Drawing('增量属性维护：结构、属性与使用者分别判断','构造基线 → 外部改图 → 调用 incremental_update',1170)
d.box(155,'构造：保存 graph 与 processed_hashes',['签名是 node、target、args/kwargs 容器身份'],0)
d.box(320,'本轮初始化',['统计 existing_storages；创建 to_process'],1)
d.box(485,'逐节点筛选',['旧签名且未标记、target 不可 fake、缺参数属性 → 右侧','否则在当前 FakeTensor mode 处理'],2)
d.box(650,'Fake 执行并比较',['被判相同 → 保留旧 metadata，直接走右侧','新签名在这个 continue 分支不会额外登记'],3)
d.box(815,'属性需要更新',['绑定符号、写 val、更新存储计数','标记使用者，保存当前签名；然后走右侧'],4,height=140)
d.arrow('M465 265 V320');d.arrow('M465 430 V485');d.arrow('M465 595 V650');d.arrow('M465 760 V815')
d.control(365,955,['跳过或更新完成','均继续下一个节点','上游标记的用户','签名未变也会处理'],460,'扫描结束：返回 None')
for y in [540,705,885]:d.arrow(f'M800 {y} H850')
d.save('incremental-properties','主要结果写入 node.meta；存储计数和已处理签名是不同状态')
d=Drawing('清理与局部顺序：按边界遍历，不跨未就绪使用者','输入为有效 FX 图；不在这里执行真实 Tensor 数值',1040)
d.box(155,'入口条件',['DCE 开关决定清理；推理且重排开关开启才继续'],0)
d.box(320,'建立扫描状态',['查首个 copy_；seen_nodes 为空','有 copy 时先跳过它及之后的尾部'],1)
d.box(485,'逆向取得当前节点',['加入 seen；尾部仍未越过 → 右侧','前缀节点继续递归访问 args / kwargs'],2)
d.box(650,'对每个输入节点运行 visit',['函数计算、非 getitem、所有用户已见、同一 region','全满足才 prepend；其余参数仍逐项检查'],3,height=140)
d.arrow('M465 265 V320');d.arrow('M465 430 V485');d.arrow('M465 595 V650')
d.control(365,790,['当前节点完成后','继续逆向下一节点','visit 只改链表','不改算子参数'],460,'扫描结束：返回 None')
for y in [540,720]:d.arrow(f'M800 {y} H850')
d.save('cleanup-order','没有 copy 时检查全图；区域号保护仍独立生效，DCE 使用另一纯度接口')
d=Drawing('删除 noop：候选、存储与谓词逐项通过才替换','输入：已有 metadata 的 graph；返回 None，原地修改图',1140)
d.box(155,'先收集接口观察',['输入节点、输入存储、输出存储；保留身份与别名区别'],0)
d.box(320,'扫描候选并取 src',['不在 registry 或源不是 FX Node → 右侧','固定参数下标，或嵌套参数 selector'],1)
d.box(485,'检查存储与身份保护',['新增不允许的输入/输出别名 → 右侧','直接 output.args 的身份条件命中 → 右侧'],2)
d.box(650,'解析参数并检查统一/专用条件',['缺 metadata、same_meta=False、cond=False → 右侧','全部满足才进入替换'],3)
d.box(815,'重接并删除',['replace_all_uses_with(src)，再 erase_node','只改 graph；调用方之后 recompile'],4)
d.arrow('M465 265 V320');d.arrow('M465 430 V485');d.arrow('M465 595 V650');d.arrow('M465 760 V815')
d.control(365,925,['拒绝、跳过或删除','都继续下一个节点','tuple 输出的身份','保护需看真实层次'],290,'扫描结束：返回 None')
for y in [375,540,705,870]:d.arrow(f'M800 {y} H850')
d.save('noop-storage','相同数值、相同 metadata、共享存储与同一 Python 对象是不同判定')
