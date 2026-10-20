"""与总图一致的模块流程图；用编号区分职责与重复调用。"""
from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent
DIAGRAMS={
'preparation':('图准备：维护算子关系和张量属性', '输入：单图 gm + example_inputs',[
('① 属性与布局描述',['view → reshape；FakeTensor 传播','返回 fake_mode；把张量属性写入节点']),
('② 图重写',['DCE、局部性重排、noop 与模式替换','可选批量融合 / 通信改写；修改同一张图']),
('④ 图收尾 · 第一次参与',['稳定拓扑排序；移动构造节点','调用①的增量更新，补回改写后的属性']),
('③ 写入语义恢复',['reinplace 与自动功能化分解','在主要功能式改写完成后引入受控写入']),
('④ 图收尾 · 第二次参与',['gm.recompile()；graph.lint()','交出可解释且顺序合法的 gm'])], '输出：整理后的 gm；fake_mode 继续供 Lowering 使用'),
'lowering':('Lowering：从算子调用构造计算与存储', '输入：整理后的 FX 图、输入属性、图上下文',[
('① 解释图与登记输入',['按 FX 节点顺序派发；建立输入布局和符号','输入节点 → TensorBox / 符号表达式']),
('② 表达计算与布局',['普通点运算、reduction、view','构造元素表达式、索引和输出尺寸']),
('④ 专用算子路径 · 按需',['模板 / 外部调用 / 控制流','与普通表达式通过 IR 输入输出衔接']),
('③ 物化与副作用',['需要存储时登记 ComputedBuffer 和 operation','输出实现、输入写回、最终布局约束'])], '输出：GraphLowering 上的 operations、buffers、graph_outputs'),
'scheduling':('调度与融合：把存储依赖变成生成顺序', '输入：graph.operations + 输入 / 常量 / 输出状态',[
('① 依赖与合法顺序',['构造调度节点；读写、别名、修改依赖','拓扑排序；删死节点；计算祖先关系']),
('② 融合与分组',['搜索合法组合；交给设备后端判断','形成融合节点；处理模板与分组节点']),
('③ 生命周期与生成驱动 · 分析',['从输出反向计算 last_usage','保留输出和必要输入存储']),
('③ 生命周期与生成驱动 · 遍历',['Scheduler.codegen 按节点调用设备后端','发出分配 / 调用 / 释放代码；flush 收尾'])], '输出：生成器中的 kernel / wrapper 内容，交代码生成封装完成'),
'codegen':('代码生成：组装并加载图的程序', '输入：调度节点、IR 布局、图输入输出和配置',[
('① 设备 kernel 与方案选择',['普通计算 / 模板 / 外部算子调用','生成后端源码；按配置选择或调优方案']),
('② Wrapper 与存储计划',['把分配、调用、释放、返回串成图入口','保留布局、别名、写入与设备语义']),
('③ 编译与加载',['提交并等待编译；缓存生成模块','JIT：module.call；AOT：编译产物路径']),
('④ 缓存与运行适配',['入口先查 FX 图缓存，命中可跳过重新生成','CompiledFxGraph：恢复程序并作运行适配'])], '返回可调用程序；之后真实输入 → wrapper → kernel / 外部调用 → 输出')}
DIAGRAMS['properties']=('属性与布局：从初次传播到改写后更新','输入：gm、example_inputs；输出仍保存在节点元数据中',[('① 初次属性建立 · 布局兼容',['view target → reshape；保持原节点身份','在属性执行前允许后续合法的布局处理']),('① 初次属性建立 · 全图传播',['复用或新建 FakeTensor mode','返回 mode；node.meta["val"] 保存属性']),('② 增量属性维护 · 改写前',['保存节点 target、args、kwargs 的身份','图改写继续复用同一张 graph']),('② 增量属性维护 · 改写后',['执行变化节点；属性改变时传播到使用者','维护符号绑定和共享存储关系'])],'交给图重写或收尾：带属性的 gm + 同一 FakeTensor mode')
import json
DIAGRAMS.update(json.loads((P/'diagram-data.json').read_text()))
colors=[('#e3f3ff','#368dc0'),('#e5f6e5','#40a147'),('#fff3c9','#dca300'),('#ffe4e4','#ef5555'),('#f0e5ff','#9959c6')]
for key,(title,inp,stages,out) in DIAGRAMS.items():
    h=310+len(stages)*180
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{h}" viewBox="0 0 1100 {h}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="{h}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
    def text(y,t,size=23): a.append(f'<text x="550" y="{y}" text-anchor="middle" font-size="{size}">{escape(t)}</text>')
    text(40,title,30);text(95,inp)
    a.append(f'<rect x="45" y="130" width="1010" height="{len(stages)*180-10}" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
    for i,(label,lines) in enumerate(stages):
        y=155+i*180;fill,stroke=colors[i%len(colors)]
        a.append(f'<rect x="110" y="{y}" width="880" height="125" rx="23" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        text(y+34,label);text(y+72,lines[0],21);text(y+104,lines[1],21)
        if i<len(stages)-1:a.append(f'<path d="M550 {y+125} V{y+178}" stroke="#222" stroke-width="3" marker-end="url(#arr)"/>')
    text(h-95,out,21)
    text(h-48,'实线：编译期间交接；同一编号重复出现表示同一模块再次参与',18)
    if key=='lowering':text(h-18,'④ 为算子分派的条件路径；普通算子直接从②交给③，并非总经过④',18)
    if key=='codegen':text(h-18,'缓存查询发生在生成前；此图展示未命中后的组装主线，缓存旁路见总图',18)
    a.append('</g></svg>');svg=''.join(a)
    (P/f'{key}.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{key}.png'))
# Lowering 的普通 / 专用算子是分派分支，不能画成必经线性阶段。
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="950" viewBox="0 0 1100 950"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="950" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
def t(x,y,s,z=21):a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}">{escape(s)}</text>')
def b(x,y,w,h,title,lines,fill,stroke):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>');t(x+w/2,y+36,title,23)
    for j,line in enumerate(lines):t(x+w/2,y+75+j*32,line,20)
def line(d,dash=False):a.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="3" marker-end="url(#arr)"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
t(550,42,'Lowering：算子分派、计算与存储交接',30);t(550,94,'输入：整理后的 FX 图 + 输入属性 + 图上下文')
a.append('<rect x="35" y="130" width="1030" height="660" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(220,160,660,125,'① 解释图与登记输入',['按节点分派；保存当前节点和 origins','placeholder → 输入布局 / 符号表达式'],'#e3f3ff','#368dc0')
b(65,365,455,170,'② 表达计算与布局',['普通点运算、reduction、view','元素 loader、逻辑索引和布局','输入或结果需要存储时调用③'],'#e5f6e5','#40a147')
b(580,365,455,170,'④ 专用算子路径',['模板、外部调用、控制流','准备专用 IR 和输入输出约束','与其他节点通过 IR 值连接'],'#ffe4e4','#ef5555')
line('M550 285 V320 H292 V365');line('M550 320 H807 V365')
t(180,345,'普通算子',18);t(920,345,'专用算子 / fallback',17)
b(220,615,660,130,'③ 物化与副作用',['登记 ComputedBuffer / operation；输出实现','布局定型、输入写回；保留图接口语义'],'#fff3c9','#dca300')
line('M292 535 V574 H425 V615');line('M807 535 V574 H675 V615')
t(550,832,'出口：GraphLowering 上的 operations、buffers、graph_outputs',21)
t(550,877,'逐节点重复分派；一个节点的 IR 结果继续作为后续节点的输入',20)
t(550,919,'实线：本层处理方向；③也可能在②或④内部因输入要求被调用',18)
a.append('</g></svg>');svg=''.join(a)
(P/'lowering.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'lowering.png'))
# 明确表示模式开关关闭时的旁路。
p=P/'rewrites.svg';svg=p.read_text()
bypass='<path d="M550 305 H85 V937 H110" fill="none" stroke="#444" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#arr)"/><text x="68" y="680" transform="rotate(-90 68 680)" font-size="17">pattern_matcher=False：跳过批量、noop 与通用模式</text>'
svg=svg.replace('</g></svg>',bypass+'</g></svg>');p.write_text(svg)
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'rewrites.png'))
# Interpreter 必须逐节点分派，不能把所有 get_attr 画成发生于所有算子之前。
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1120" viewBox="0 0 1100 1120"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="1120" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
t(550,42,'解释图：逐节点分派，保留同一图上下文',30)
t(550,95,'输入：gm、example_inputs、可选 ShapeEnv')
a.append('<rect x="30" y="130" width="1040" height="780" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
b(220,155,660,115,'① 图上下文与符号状态',['建立 GraphLowering 和符号 / 输入 / 操作映射','安装 V.graph、V.fake_mode'], '#e3f3ff','#368dc0')
line('M550 270 V305')
a.append('<path d="M550 305 L675 350 L550 395 L425 350 Z" fill="#fff3c9" stroke="#dca300" stroke-width="3"/>')
t(550,357,'下一个 FX 节点',20)
b(60,450,455,150,'② 输入和常量登记',['placeholder → 输入 IR / 符号','get_attr → Constant / Buffer / 子图','结果放回当前节点环境'],'#e5f6e5','#40a147')
b(585,450,455,150,'③ 节点派发与结果维护 · 派发',['call_function 调相邻 lowering 模块','output 调相邻物化与副作用模块','在当前节点与 origins 作用域处理'],'#ffe4e4','#ef5555')
line('M425 350 H287 V450');line('M675 350 H812 V450')
t(140,420,'placeholder / get_attr',18);t(930,420,'函数 / output 等',18)
b(220,670,660,140,'③ 节点派发与结果维护 · 收尾',['满足布局、物化与复用要求；登记来源与读者','处理新增符号与断言；result 留给后续节点'],'#f0e5ff','#9959c6')
line('M287 600 V630 H425 V670');line('M812 600 V630 H675 V670')
line('M880 740 H1055 V290 H550');t(969,325,'继续处理节点',17)
line('M550 810 V960');t(760,875,'全部节点处理完成',19)
b(120,960,860,95,'出口：节点环境中的 IR 值和图上状态',['graph_inputs、constants、operations、graph_outputs'],'#fafafa','#666')
t(550,1090,'实线为实际处理方向；右侧独立通道表示逐节点循环',18)
a.append('</g></svg>');svg=''.join(a)
(P/'interpretation.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/'interpretation.png'))
# 三类普通 / 专用算子按语义分派；物化也是由不同需求触发，均避免伪线性流程。
three_way_data={
'expressions':('表达计算与布局：给索引，取得值或映射', '输入：当前算子 target、IR 输入与参数',[
('① 点表达式与类型处理',['按规则提升 dtype / 处理标量','取得输入 loader 与 ranges','组合为 Pointwise.inner_fn','广播时调用②']),
('② View、广播与布局索引',['重排逻辑 index 或建立新 layout','expand 的复用维 stride 可为 0','已有存储可建 ReinterpretView','继续引用原输入或表达式']),
('③ 归约与扫描',['分开输出维与归约 / 扫描维','组合 index 与 reduction_index','选择展开或结构化的计算 IR','调用输入 loader 和布局机制'])],
'输出：Pointwise / View / Reduction / Scan 等 IR；有存储要求时交物化模块',
'三个分支按算子语义选择；②的索引机制也被其他分支调用'),
'materialization':('物化与副作用：三种需求共享存储实现', '输入：表达式 / 视图，及输出、复用或写入要求',[
('① 存储实现与复用提示',['StorageBox.realize / mark_reuse','已有存储返回其名字','未实现计算登记 ComputedBuffer','保留来源；布局可暂为 Flexible']),
('② 输出与布局约束',['实现可返回的输出形式','保留 stride 和共享关系要求','需要时调用①取得稳定存储','登记 graph_outputs 并定型布局']),
('③ 写入与旧值保护',['检查内部临时量或图输入','先调用①保护必要旧值读取','建立指向目标的 MutationLayout','输入写回保留实际存储身份'])],
'输出：更新后的 graph.buffers / operations / outputs，交 Scheduler 处理',
'分支是不同触发需求，调用可交错；登记 IR 不等于现在分配真实 Tensor'),
'special-ops':('专用算子：调用协议、模板或结构化子图', '输入：专用算子 target、IR 参数和图上下文',[
('① 外部调用与 fallback 协议',['实现输入；Fake 推导输出描述','建立 packed 调用与输出路径','记录 schema、别名和修改','MultiOutput 可共享同一次调用']),
('② 模板与复合算子接入',['按设备、形状和配置准备方案','连接模板、外部实现与布局','选择器细节归代码生成模块','外部方案使用①对应协议']),
('③ 条件与循环子图',['为分支或 cond/body 降低子图','核对父图与子图的输出接口','登记 Conditional / WhileLoop','保留 MultiOutput 与复用限制'])],
'输出：专用操作和其 IR 输出，交回普通节点环境，继续整张图编译',
'编译子图不等于立即执行子图；运行时再选择分支或重复循环')}
three_way_data.update(json.loads((P/'backend-flow-data.json').read_text()))
for key,(title,inp,branches,out,note) in three_way_data.items():
    a=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="880" viewBox="0 0 1100 880"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="880" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
    t(550,42,title,29);t(550,95,inp,22)
    a.append('<rect x="25" y="135" width="1050" height="500" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
    b(270,165,560,100,'按当前需求进入相应处理',['分支不是三个连续的必经步骤'],'#fafafa','#666')
    a.append('<path d="M550 265 V300 H205 V355 M550 300 V355 M550 300 H895 V355" fill="none" stroke="#222" stroke-width="3"/>')
    for i,(label,lines_) in enumerate(branches):
        x=45+i*345;fill,stroke=colors[i]
        a.append(f'<rect x="{x}" y="355" width="320" height="215" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        t(x+160,391,label,19)
        for j,txt in enumerate(lines_):t(x+160,430+j*31,txt,17)
        line(f'M{x+160} 320 V355')
    a.append('<path d="M205 570 V610 H550 M550 570 V610 M895 570 V610 H550" fill="none" stroke="#222" stroke-width="3"/>')
    line('M550 610 V680')
    b(60,680,980,95,'交回父图 Lowering 或后续编译阶段',[out],'#f0e5ff','#9959c6')
    t(550,825,note,18);t(550,860,'实线为本层交接；同一职责可在一次图遍历中多次参与',17)
    a.append('</g></svg>');svg=''.join(a)
    (P/f'{key}.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{key}.png'))
# 融合和 codegen 的迭代使用独立右侧返回通道。
for key,yfrom,yto,label in [('fusion',577,155,'数量变化时继续下一轮'),('lifetime',577,335,'按需处理释放后继续下一节点')]:
    f=P/f'{key}.svg';svg=f.read_text()
    extra=f'<path d="M990 {yfrom} H1030 V{yto-15} H550 V{yto}" fill="none" stroke="#444" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#arr)"/><text x="1010" y="580" transform="rotate(-90 1010 580)" font-size="16">{label}</text>'
    svg=svg.replace('</g></svg>',extra+'</g></svg>');f.write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{key}.png'))
def begin_flow(height,title,inp,inner_height):
    global a
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs><rect width="1100" height="{height}" fill="white"/><g font-family="Noto Sans CJK SC,sans-serif" fill="#222">']
    t(550,42,title,29);t(550,95,inp,21)
    a.append(f'<rect x="30" y="130" width="1040" height="{inner_height}" rx="35" fill="#fff9d9" stroke="#d9a300" stroke-width="3"/>')
def end_flow(key):
    a.append('</g></svg>');svg=''.join(a)
    (P/f'{key}.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{key}.png'))
begin_flow(1160,'编译装载：JIT 入口与 AOT 产物两条出口','输入：生成源码、参数类型、行映射与编译模式',850)
b(220,155,660,110,'③ JIT 与 AOT 装载 · 选择出口',['JIT 加载当前进程模块；AOT 返回编译产物路径'],'#f0e5ff','#9959c6')
line('M550 265 V300 H295 V350');line('M550 300 H820 V350')
t(190,330,'JIT',19);t(930,330,'AOT',19)
b(60,350,470,125,'② 源码和二进制缓存',['写 Python 源码；按 key/path 加载模块','后端二进制也由相应缓存服务'],'#e5f6e5','#40a147')
b(600,350,440,240,'③ AOT 编译产物出口',['要求相应 C++ wrapper','处理外部节点序列化等参数','由 AotCodeCompiler 编译产物','交出路径，不返回 Python call'],'#f0e5ff','#9959c6')
line('M295 475 V530')
b(60,530,470,140,'① 编译任务与等待',['模块顶层提交 kernel 编译 / 加载','Future 或函数进入 globals','wait 取得可调用 kernel'],'#e3f3ff','#368dc0')
line('M295 670 V740')
b(60,740,470,125,'③ JIT 与 AOT 装载 · JIT 返回',['安装常量和行映射；取得 module.call','交给 CompiledFxGraph 封装'],'#f0e5ff','#9959c6')
line('M295 865 V990 H425 V1010');line('M820 590 V990 H675 V1010')
b(120,1010,860,95,'交回对应出口类型',['JIT：可调用入口；AOT：编译产物路径'],'#fafafa','#666')
t(550,1140,'①调用②编译后端文件；图展示出口差别，AOT 不是 JIT 的继续执行',18)
end_flow('build-load')
begin_flow(1170,'缓存：先判断适用性，再恢复或正常编译','输入：gm、编译示例、配置和当前 tracing context',910)
b(220,155,660,120,'① 键与候选适用性',['能否缓存、图与输入描述、配置、候选 guards','查询结果为 hit、miss 或 bypass'],'#e3f3ff','#368dc0')
line('M550 275 V330 H287 V440');line('M550 330 H812 V440')
t(170,395,'hit',19);t(930,395,'miss / bypass',19)
b(60,440,455,145,'② 恢复与持久化 · 恢复',['反序列化程序描述；检查适用条件','加载源码模块，建立 current_callable','使用本进程入口，不是保存的函数地址'],'#e5f6e5','#40a147')
b(585,440,455,145,'共享协作：正常单图编译',['实际经过图准备、Lowering、调度与生成','由本系列其他模块交回 CompiledFxGraph'],'#fafafa','#666')
line('M812 585 V650')
b(585,650,455,120,'② 恢复与持久化 · 保存',['可缓存 miss 保存源码、元数据和 guards','bypass 不按这条普通保存路径缓存'],'#e5f6e5','#40a147')
line('M287 585 V820 H425 V870');line('M812 770 V820 H675 V870')
b(220,870,660,140,'③ 调用适配与 CUDA Graph',['回填 stride；按条件捕获 / 禁用 CUDA Graph','处理输入对齐；返回当前环境可调用图'],'#f0e5ff','#9959c6')
t(550,1090,'之后才用真实输入调用：hit / miss / bypass 都计算本次结果',21)
t(550,1140,'实线为编译 / 恢复期间交接；共享编译流程保留原模块归属',18)
end_flow('cache-runtime')
begin_flow(1200,'初次属性建立：完整函数的分支与副作用','输入：gm、example_inputs、force_allow_non_fake_inputs',1000)
b(220,155,660,110,'view_to_reshape',['查找 aten.view.default，修改同一节点的 target','不复制 Tensor，也不在这里 recompile'],'#e3f3ff','#368dc0')
line('M550 265 V320')
b(220,320,660,110,'fake_tensor_prop：检测 mode',['当前 context / 活动 mode / 输入必须一致','没有现成 mode 才创建'],'#fff3c9','#dca300')
line('M550 430 V470 H287 V520');line('M550 470 H812 V520')
t(175,495,'没有 mode',18);t(930,495,'已有 mode',18)
b(60,520,455,180,'创建并转换参数',['新 mode：allow_non_fake_inputs=True','FakeTensorProp.propagate 转换输入','传播结果写入 node.meta["val"]'],'#e5f6e5','#40a147')
b(585,520,455,180,'复用并按条件临时放宽',['force=False：nullcontext','force=True：临时把 allow flag 设为 True','propagate_dont_convert_inputs'],'#ffe4e4','#ef5555')
line('M287 700 V750 H425 V810');line('M812 700 V750 H675 V810')
b(220,810,660,145,'成功出口',['返回 fake_mode；属性仍保存在 gm 节点中','复用分支的临时 flag 在退出上下文后恢复'],'#f0e5ff','#9959c6')
a.append('<path d="M220 375 H45 V1070 H220" fill="none" stroke="#555" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#arr)"/>')
b(220,1020,660,100,'检测到不同 mode：直接抛 AssertionError',['force=True 不合并环境；不会进入后面的传播'],'#fafafa','#666')
t(550,1160,'实线为成功路径；虚线为检测冲突出口。传播自身失败也会向调用者抛出',18)
end_flow('initial-properties')
