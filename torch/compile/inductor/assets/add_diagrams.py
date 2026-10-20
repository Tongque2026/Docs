"""扩展流程图数据；不含源码行为的替代实现。"""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
items={
'rewrites':('图重写：从候选关系到合法替换','输入：带 FakeTensor 属性的单图 gm',[
('① 清理与局部顺序 · 按配置',['DCE 与推理局部性重排','保留图接口和写入区域约束']),
('④ 批量计算与通信改写 · 批量部分',['pattern matcher 开启时按规则分组','没有配置或候选则保持图不变']),
('② 无效操作与存储边界',['检查 metadata、算子条件、输入输出别名','重接可删除节点的使用者']),
('③ 模式匹配与子图替换',['按根 target 检索；匹配和 extra_check','检查 mutation region 后替换或插入 handler']),
('④ 批量计算与通信改写 · 通信部分',['按各自开关应用微流水 / DDP 等处理','交给共享属性维护和图收尾'])],
'输出：同一 gm 上新的算子关系；不是 kernel 或结果 Tensor'),
'mutation':('写入恢复：合法性判断先于删 clone 和 copy','输入：主要重写完成并补齐属性的图',[
('① View/scatter 表示与分解 · 统一表示',['记录 base 和有序 view 链','构造可统一分析的 generalized scatter']),
('② 存储可写性与原地化',['检查共享存储、后续旧值读取、输入回写意图','改 target / 引用，或写入保留 clone 清单']),
('① View/scatter 表示与分解 · 恢复操作',['根据判定选择功能式或写入式分解','交回正常的 view / scatter / copy 节点']),
('③ 自动功能化包装分解',['按 clone 清单展开高阶包装','移除 auto_functionalized / Triton functional wrapper'])],
'输出：保留正确输入修改与别名的图；再交图收尾'),
'finalization':('图收尾：排序、设备、属性与入口一致性','输入：图重写后的 graph 和原 GraphModule',[
('① 顺序与入口一致性 · 排序',['pending / ready / waiting 保证生产者先就绪','发现数据环或未处理节点则断言失败']),
('② 构造节点设备迁移',['检查唯一目标设备与完整使用链','只修改允许迁移的 CPU constructor device']),
('共享协作：属性与布局描述',['调用原 FakeTensorUpdater 补回变化属性','本模块不复制属性维护实现']),
('相邻协作：写入语义恢复',['reinplace、包装分解与通信写入处理','图返回后再生成最终 Python 入口']),
('① 顺序与入口一致性 · 入口和校验',['gm.recompile 对齐 graph 与 forward','graph.lint 检查结构；不是数值证明'])],
'输出：可交给 GraphLowering 的 gm 和一致属性状态'),
'interpretation':('解释图：节点值进入同一次编译的 IR 环境','输入：gm、example_inputs、可选 ShapeEnv',[
('① 图上下文与符号状态',['建立 GraphLowering；保留符号、输入、缓冲区状态','安装 V.graph / V.fake_mode 的线程局部作用域']),
('② 输入和常量登记 · 遇到相应节点',['placeholder → InputBuffer / 符号表达式','get_attr → 常量、标量或子图 IR']),
('③ 节点派发与结果维护 · 调用前半',['取前驱 IR 值；安装当前节点和 origins','处理特殊布局 / fallback / 符号分支']),
('相邻协作：表达式与专用算子',['普通元素表达式或模板 / 外部调用','返回 IR 句柄；可能登记图操作']),
('③ 节点派发与结果维护 · 返回后半',['根据使用者满足布局、物化和复用要求','登记来源、读者和符号绑定，再处理下一个节点'])],
'出口：节点环境中的 IR 值，加 GraphLowering 上的持久图状态')}
(P/'diagram-data.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
f=P/'modules.py';s=f.read_text()
if "diagram-data.json" not in s:
    s=s.replace('colors=[',"import json\nDIAGRAMS.update(json.loads((P/'diagram-data.json').read_text()))\ncolors=[",1)
f.write_text(s)
