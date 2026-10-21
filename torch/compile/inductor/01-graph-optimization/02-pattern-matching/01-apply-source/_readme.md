# Q121 作者记录：`PatternMatcherPass.apply`

> 第五轮状态：叶子完成。重新对照当前 `pattern_matcher.py` 声明范围，并以分支状态和迁移计数复核，不用 AST/行数替代教学验收。

## 读者、层次与目标

- 所在层：深度 3，Q12 的第一个实现叶子；不为 `apply` 内每个条件另建目录。
- 已有前提：读者已在 Q12 看过 pattern、Match、mutation region、extra check 和 replacement 的本层协作。
- 本叶子职责：从注册表构造候选锚点序列，逐个执行类型、结构、region 和附加条件检查，再调用 entry 并计数。
- 入口：有注册规则的 `PatternMatcherPass` 与 `GraphModule` 或其 `Graph`。
- 出口：原图可能被 entry 修改，返回成功应用 entry 的次数；非法输入或 entry 内异常是失败出口。
- 承诺深度：`__init__`、`__getitem__`、`apply` 全部源码及每个判断、循环、状态和正常/拒绝出口的推演。

## 声明范围与证据边界

- 产品文件：`/work/pytorch/torch/_inductor/pattern_matcher.py:1660-1733`。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 当前范围从 `class PatternMatcherPass` 的 `__init__` 开始，到 `apply` 的 `return count`；紧随其后的 `clear` 不属于本叶子声明范围。
- 资料：当前源码、当前 `test/inductor/test_pattern_matcher.py` 的 mutation 等测试、独立实验 `../../../experiments/02_pattern_matcher.py` 与产物。
- Q122 承担 `ReplacementPatternEntry.replace_with_graph`；本篇只追到 `entry.apply` 的接口与外部可观察结果。

## 篇内知识点队列

| 顺序·编号 | 问题、状态与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | 解释空表与三种输入类型路径，能指出快速出口先于类型检查 | empty/object、GraphModule、Graph | 完成并复读 |
| 2·K02 | 从 registry key 构造普通与 call_module 候选集合 | mul 锚点、ReLU+Sigmoid 模块图 | 完成并复读 |
| 3·K03 | 按源码顺序推演 unsupported、match、region、debug、extra_check | float32 成功及六种跳过路径 | 完成并复读 |
| 4·K04 | 解释逆序、entry 注册顺序、`_erased`、local/global counters | 双 neg、双 entry 擦除 | 完成并复读 |
| 5·K05 | 用改变 dtype/锚点数量的迁移例独立算出三个计数 | 两个匹配、一个属性拒绝 | 完成并复读 |

当前点：K05 完成；篇内无剩余项。根队列下一项为同层 Q122。

## 处理情况覆盖

| 情况 | 关键状态或出口 | 正文位置 | 直接证据 | 状态 |
| --- | --- | --- | --- | --- |
| 空注册表 | 在读图和类型检查前返回 0 | “入口分支” | 实验传 `object()` | 叶子机制完成 |
| GraphModule | 使用 `.graph` | “入口分支” | 主实验 | 叶子机制完成 |
| Graph | 使用自身与 owning module | “入口分支” | 实验匹配 1 次 | 叶子机制完成 |
| 非法类型 | 非空 matcher 抛 RuntimeError | “入口分支” | 实验错误文本 | 叶子机制完成 |
| region 尚未计算 | 为全图节点写 mutation_region_id | 成功表与 mutation 案例 | 实际 meta 输出 | 叶子机制完成 |
| 普通 op/target | 精确 `find_nodes` | “候选收集” | mul/no-anchor 案例 | 叶子机制完成 |
| call_module | 先收全体，再按实例类过滤 | “候选收集” | ReLU/Sigmoid 实验 | 叶子机制完成 |
| unsupported type | entry.match 前 continue | “六种不应用” | disable_cpp_codegen CPU 输出 | 叶子机制完成 |
| 结构失败 | `is_match=False`，不检查 region/extra | 常数 3 | 独立实验 | 叶子机制完成 |
| 跨 mutation | Match 成功但 region 集合不唯一 | add/copy_/mul | 独立实验 | 叶子机制完成 |
| extra check 拒绝 | int64 结构成功，属性门失败 | int64 案例 | 图与 counters | 叶子机制完成 |
| debug 名字命中 | 写日志后继续正常判定 | “六种不应用” | 当前源码；未采集日志 | 源码机制完成，证据边界已标 |
| 多候选锚点 | FX 逆序处理 | 双 neg | 实际事件顺序 | 叶子机制完成 |
| 同锚点多 entry | 注册顺序；擦除后 break | 双 entry neg | 实际事件仅 first | 叶子机制完成 |
| 成功计数 | local 按 entry；global 按 entry/nodes | 主例、双 neg | counters 实测 | 叶子机制完成 |

若 `entry.apply` 自身抛异常，源码没有在本函数捕获；local count 已在调用前递增，而两个全局 counter 位于调用后。函数异常退出且没有正常返回，正文不拿这个中间状态定义成功统计。本教程主线只解释正常完成的 entry。

## 实验事实

原实验已有：float32 成功；int64 extra check 拒绝；常数 3 结构失败；共享 add 仍成功且保留原 add；跨 copy_ region 拒绝；无 mul 锚点；重复 PatternExpr 身份约束。

本轮扩展并重跑后新增：

```text
empty matcher + object input -> 0
nonempty matcher + object input -> RuntimeError
Graph input -> 1
disable_cpp_codegen=True, CPU output -> 0
two neg anchors -> order ['neg_1', 'neg'], count 2, counters 2/2
two entries, first erases -> events ['first'], count 1
call_module ReLU+Sigmoid, only ReLU registered -> events ['relu'], count 1
float32 add+mul -> counters count=1, nodes=2
int64 -> counters empty
```

CPU fallback 实验核对的是当前 helper 在当前配置下的实际路径，不推广到所有设备/类型。call_module 实验使用 `torch.fx.symbolic_trace` 的真实 module nodes，但 handler 只记录事件，不修改图。双 neg 与擦除实验使用当前 `GraphPatternEntry` 接口，产品 `PatternMatcherPass.apply` 未被替换。

## 源码完整性审计

- 构造器的类型、`defaultdict(list)` 和 pass name 均保留。
- `__getitem__` 全部保留。
- apply 的空表、GraphModule/Graph/else、region 初始化、partial、三个局部状态、两种候选收集和 observer 均保留。
- node 循环中的 extract target、call_module 过滤、fallback、entry 循环、`_erased`、match、mutation barrier、debug、extra check、entry apply、两个 counters 与最终 return 均保留。
- 教学注释统一用 `# 教学：`；产品三行 CPU 注释及 mutation 注释原样保留；无省略号。

AST 等价检查需在每次正文修改后重新执行。源码无漏行只能证明范围完整，不能替代上面的分支状态与迁移验收。

## 只凭正文的作者理解验收

1. **为什么空 matcher 接受 `object()`，非空 matcher 却报错？** 空表 return 位于类型分支之前；非空才会落入 GraphModule/Graph/else。
2. **为什么注册 nn.ReLU 后不能直接按 target `"relu"` 查？** 属性名随 GraphModule 命名而变；注册键使用模块类，所以先取所有 call_module，再从 owning module 提取实例类。
3. **`(x+y)*3` 与 int64 `(x+y)*2` 分别在哪退出？** 前者 pattern match 因常数失败；后者 Match 和 region 成功，在 extra_check 失败。
4. **跨 copy_ 为什么不调用 extra_check？** mutation region 检查位于 debug/最终成功条件之前，集合有两个 id 时直接 continue。
5. **两个 neg 为什么顺序是 `neg_1,neg`？** 候选摊平后按 FX Node 排序并 reverse；这不是注册表顺序。
6. **第一条 entry 擦除锚点后第二条怎样退出？** 回到同一 entry 循环顶部，`node._erased` 为 true，于是 break。
7. **一个 float32、一个 int64 的两个 add→mul，三个增量是多少？** local count 1、global match count 1、matched nodes 2；失败候选不进入计数。
8. **为什么生成的三个 replacement 节点不让 nodes counter 加 3？** counter 使用应用前 `len(m.nodes)`，本例 Match 只有原 add 与 mul。

答案均可从正文的具体分支、状态表和实验结果推出。实际读者反馈仍未取得；这里只记录作者复读。

## 本轮整改与续写

原稿保留了源码，但作者记录只有四条完成声明；正文实际实验不足以支撑空表、输入类型、call_module、fallback、逆序双锚点、擦除与 counters 等完整路径。本轮先扩展实验，再重建层次、入口/出口、覆盖、证据边界和迁移题。还按当前 helper 修正了产品注释可能造成的“所有 CPU 输入都跳过”误读。

## 第五轮叶子验收

- **完整源码**：`__init__`、`__getitem__`、`apply` 的连续声明范围无省略，产品注释保留，教学注释均有固定标记；`clear` 明确不在范围内。
- **分支与出口**：空表、GraphModule/Graph/错误类型、region 初始化、普通与 call_module 候选、unsupported、结构失败、跨 mutation、debug、extra_check、`_erased`、entry 调用、局部/全局计数与异常边界均有归属。
- **具体状态**：float32 成功表逐门推进；六条拒绝路径分别指出退出位置；双 neg 给出逆序状态；双 entry 给出擦除后的 break；call_module 给出 ReLU/Sigmoid 类过滤。
- **理解迁移**：正文可独立算出“两项中后一项 int64”的 local/global/nodes 增量为 1/1/2，也能解释空 matcher 为何接收非法对象、跨 mutation 为何不会调用 extra_check。
- **证据边界**：日志分支由源码支持；call_module handler 只记录事件；`entry.apply` 内部替换机制不在本篇冒充完成，继续由 Q122 承担。

Q121 第五轮叶子复核完成。下一项 Q122：`ReplacementPatternEntry.replace_with_graph` 的完整源码与具体节点映射状态。
