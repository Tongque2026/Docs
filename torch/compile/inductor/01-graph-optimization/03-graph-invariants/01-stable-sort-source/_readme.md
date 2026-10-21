# Q131 作者记录：`stable_topological_sort`

> 第五轮状态：叶子完成。重新以完整函数、容器不变量、逐轮状态、非法输入与迁移推演验收。

## 读者、层次与目标

- 所在层：深度 3，Q13 的第一个实现叶子。
- 读者前提：已从 Q13 知道图改写后需要恢复拓扑、meta、生成代码与死节点等不同不变量；会读 FX Node 的 args/users。
- 本叶子职责：收集每个节点引用的输入，移动现有节点形成稳定拓扑序。
- 入口：一张可能线性顺序非法、但预期依赖来自同一 graph 且无环的 FX Graph。
- 正常出口：所有 Node 进入 ready，waiting 为空；节点身份、target、args、meta 不因本函数重建。
- 失败出口：环或本图无法满足的外部/缺失依赖使最终断言失败。
- 承诺深度：`_args` 与 `stable_topological_sort` 全部源码，以及每轮 pending/ready/waiting/cursor 的具体推演。

## 声明范围与资料

- 产品文件：`/work/pytorch/torch/_inductor/pattern_matcher.py:1893-1935`。
- 完整范围：连续函数 `_args` 与 `stable_topological_sort`。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 资料：当前源码、当前 `test/inductor/test_pattern_matcher.py:test_stable_topological_sort`、独立实验 `../../../experiments/03_graph_invariants.py` 及产物。
- fake meta、recompile、lint、DCE 只作为职责边界对照；其内部不属于本叶子。

## 篇内知识点队列

| 顺序·编号 | 问题、状态与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | `_args` 怎样穿过 args/kwargs 容器且保留重复 Node | 嵌套参数 `[x,y,x,y]` | 完成并复读 |
| 2·K02 | 建立 pending/ready/waiting/cursor 的不变量 | 三容器表 | 完成并复读 |
| 3·K03 | 逐轮推演一次消费者早于输入的修复 | `y,add,x` 完整表 | 完成并复读 |
| 4·K04 | 解释最后缺失输入、再次挂起与多 waiter 稳定性 | checks=3、neg/relu | 完成并复读 |
| 5·K05 | 推演深依赖链、已合法图、空图、外部依赖和环 | 五组独立图 | 完成并复读 |
| 6·K06 | 区分 sort 与 meta/lint/recompile/DCE | target 修改、meta 与 DCE 对照 | 完成并复读 |

当前点：K06 完成；篇内无剩余。根队列下一项 Q132。

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文位置 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| args/kwargs 嵌套 | map_arg 依次收 Node，保留重复 | `_args` 节 | 实际 `[x,y,x,y]` | 叶子机制完成 |
| 已合法图 | cursor.next 即当前节点，不移动 | “已合法图” | 当前测试 + 实验 b,a,add | 叶子机制完成 |
| 单个消费者提前 | 挂最后缺失输入，唤醒后移动 | y/add/x 表 | lint 前失败、后通过 | 叶子机制完成 |
| 唤醒时仍缺其他输入 | 重新计算并改挂 | last-arg 节 | add 实际检查 3 次 | 叶子机制完成 |
| 同 dependency 多 waiter | reversed+stack 保留原次序 | neg/relu | 实际 x,neg,relu | 叶子机制完成 |
| 深层倒序链 | ready 逐级唤醒 | output→mul→add | 实际最终序列 | 叶子机制完成 |
| 空图 | while 零次、断言通过 | 边界节 | 实际 ok | 叶子机制完成 |
| 外部依赖 | waiting key 永不 ready | 边界节 | 实际 AssertionError | 叶子机制完成 |
| 环 | 所有节点等待，无初始 ready | 边界节 | 实际 AssertionError | 叶子机制完成 |
| 其他收尾 | 不改 meta/code/DCE | 边界节 | 独立对照实验 | 边界完成 |

## 实验事实

原实验只覆盖 `y,add,x → y,x,add` 与 lint。扩展后实际输出：

```text
stable_independent_order=b,a,add
deep_broken_order=output,mul,add,x,y
deep_fixed_order=x,y,add,mul,output
same_dependency_waiters=x,neg,relu
last_arg_recheck=y,x,add checks=3
nested_args=x,y,x,y
external_dependency=AssertionError
cycle=AssertionError
empty_graph=ok
```

`checks=3` 通过临时包装产品模块的 `_args` 记录调用次数；排序主体仍调用当前产品 `stable_topological_sort`。这项观察直接推翻原稿“最后缺失输入 ready 时其他输入应都 ready”的过强解释，并证明重新检查是安全机制的一部分。

深链与非法图只验证节点顺序/断言，不执行其 target。recompile、FakeTensorUpdater 和 DCE 沿用同一实验的独立部分，分别证明职责边界，不能计作 stable sort 的功能。

## 源码完整性审计

- `_args` 的类型、局部列表、`map_arg((args,kwargs), append)` 与返回完整。
- stable sort 的全部产品说明注释、pending/ready/waiting/cursor 初始化完整。
- while、waiting_for、最后缺失输入、ready、条件移动、cursor 更新、waiting 唤醒和最终复合断言完整。
- 教学注释统一 `# 教学：`；未省略或改写产品语句。

每次正文修改后以 AST 对照两个函数。完整性不能替代状态推演和失败边界验收。

## 只凭正文的作者理解验收

1. **为什么 pending 初始反转？** 循环从列表末端 pop；反转后第一个 pop 是当前图首节点，才能以原次序建立稳定前缀。
2. **`add,y,x` 且 args=(x,y) 为什么 add 检查三次？** 初查挂 y；y 醒来时 x 未 ready，改挂 x；x 醒来后第三次才无缺失。
3. **两个 waiter 为什么不会因栈 pop 颠倒？** waiting 保持 append 顺序；唤醒时先 reversed 再 extend，末端 pop 恢复原顺序。
4. **`_args` 返回重复 x 会让 x 必须 ready 两次吗？** 不会；ready 是集合，两份 x 引用都用同一个成员判断；x ready 后两项同时从 waiting_for 消失。
5. **外部 Node 为什么不能被算法顺手加入 ready？** pending 只由 graph.nodes 和 waiting 唤醒项构成；外部 Node 从未作为本图待处理节点出现。
6. **为什么已合法 b,a,add 不改成 a,b,add？** b/a 均按原顺序 ready 且紧邻 cursor；算法没有按参数或名字排序。
7. **stable sort 后是否可省略 recompile？** 不可由本函数推出；它只移动 graph Node，GraphModule.forward 的代码更新是另一操作。
8. **环失败前是否保证图未动？** 不保证；算法非事务，其他已 ready 节点可能已移动，最终断言只检查完成性。

答案均可由正文推演，不依赖作者脑中未写的实现。实际读者反馈尚无。

## 本轮整改与续写

原稿保留源码，但只有单一三节点案例，并错误地把“最后缺失输入”解释成任意乱序图都只重查一次；缺少嵌套 args、多 waiter、深链、空图、外部依赖、环和非事务边界。本轮扩展实验并据实重写计划、正文与理解验收。

## 第五轮叶子验收

- **完整源码**：`_args` 与 `stable_topological_sort` 连续范围全部保留，产品注释和教学注释可区分；未把排序替换成另一种拓扑算法。
- **分支与出口**：合法图、消费者提前、再次挂起、多 waiter、深链、重复参数、空图、外部依赖和环都落到 pending/ready/waiting/cursor 及最终断言。
- **具体状态**：`y,add,x` 有四轮完整表；`add,y,x` 实际重查三次；同一 x 的 neg/relu waiter 解释 reversed+stack；深链展示逐级唤醒。
- **理解迁移**：正文可独立推出两个 `(x,y)` 消费者两次改挂后仍保持原序，也能判断 consumer2 新增 consumer1 依赖时还需第三次等待。
- **职责边界**：排序只移动现有 Node；meta、recompile、lint 和 DCE 各有对照但不算作本函数效果。失败也不保证回滚已移动节点。

Q131 第五轮叶子复核完成。下一项 Q132：FakeTensorUpdater 的增量 meta 更新完整源码与传播状态。
