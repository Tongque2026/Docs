# 计算 IR 篇作者记录

## 所在层、输入与出口

- 队列位置：Q22，第 2 层，父项 Q20；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q221 叶子的源码回修。
- 读者前置：Q21 已说明 FX node 如何经 lowering 返回 TensorBox/IR；读者知道逻辑 shape/stride，但尚不会从 IR 推导每个位置读什么。
- 本篇输入：Lowering 已规范化的 tensor 参数、device/dtype、逻辑 size、axis/dims，以及输入 loader/view。
- 本篇出口：Pointwise 的输出 ranges/inner_fn，Reduction 的输出 ranges/reduction_ranges/type/loader，或 View 的新 size 与 reindex/layout；Scheduler/Codegen 据此获得迭代空间、访问规则和共享关系。
- 直接子职责：Pointwise 逐输出位置计算；Reduction 为每个输出组合一组输入；View 只改变新坐标到原数据的映射。三者紧密共享 loader/index 语言，在本篇作为同层职责协作；Q221 下一层叶子再展开完整构造源码。
- 本层不讲 realize/layout 决策和 alias/mutation，它们由同层下一篇 Q23 接住；不提前展开 Q221 的 split reduction helper 每个分支。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 用具体 `[2,3]` 输入逐位置推出 Pointwise ranges、loaders 和 inner_fn | 同形状 add 六位置 | 完成，已实测并复读 |
| 2·K02-R | 解释广播由输入 view/reindex 改写读取位置，而外层输出 index 不变 | `[2,3]+[1,3]` | 完成，已实测 stride `[0,1]` |
| 3·K03-R | 区分输出坐标与组内坐标，推演 axis/keepdim，并用非展开例证明 Reduction IR | `[2,3]` 数学 + `[2,128]` 两种实际 IR | 完成，已实测并复读 |
| 4·K04-R | 比较 ReinterpretView 与保留 reindexer 的 view，推导物理槽位 | `[2,3]→[3,2]` permute | 完成，已核对边界 |
| 5·K05-R | 将 View→Pointwise→Reduction 完整走到本层出口 | permute→add/relu→sum，接 mm 前停止 | 完成，实验 17 已整合 |
| 6·K06-R | 完成改变广播轴、规约轴、view dims 的迁移题 | 只凭正文计算新位置与出口 IR | 完成，已复读 |
| 7·K07-R | 区分零项恒等值、零项不支持、单项直接读取/索引和小静态展开 | `Reduction.create` 直接实验与具体 `[2,0]`/`[2,1]` 推演 | 完成，已实测并复读 |

## 处理情况与案例覆盖

| 情况与条件 | 索引/状态变化 | 本层出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| 同 shape Pointwise | 同一输出 index 交给各输入 loader | Pointwise ranges 等于输出 shape | 实验 05 `[2,3]` add | 已实测并讲解 |
| size-1 广播 | 被广播输入的对应 index 清零 | 外层 ranges 不变；storage-backed 输入成为 stride 0 ReinterpretView | 实验 05 broadcast | 已实测并讲解 |
| 标量/常量参与 Pointwise | tensor loader 与编译期常量共同进入 fn | 同一 Pointwise inner_fn | add/relu/mul 链 + 当前源码 | 已讲机制 |
| 普通 Reduction | kept/reduced dims 分离 | Reduction ranges + reduction_ranges | 实验 05 `[2,128]` sum | 已实测并讲解 |
| keepdim | reduced 维在输出中保留 size 1，但不加入组内选择 | 输出 ranges `[2,1]`；reduction_ranges 仍 `[128]` | 实验 05 keepdim | 已实测并讲解 |
| 小/空/特殊 reduction | 0 项按支持类型产生恒等值或拒绝；1 项直接读取/索引；小静态范围可展开 | Pointwise 或明确异常，不保证保留 Reduction | 当前源码与实验 05 | 机制已讲并复读 |
| storage-backed permute | size/stride 可直接表示 | ReinterpretView，共享 InputBuffer，0 operation | 实验 05 | 已实测并讲解 |
| 未固定 storage 的 view | 新 index 经 reindexer 转回旧 index | View/loader 组合，不必立即新增 operation | 当前源码与组合推演 | 已讲机制，待复读 |
| View→Pointwise→Reduction 综合 | view 换坐标，pointwise 加算术，reduction 增组内坐标 | ComputedBuffer Reduction；后续 mm 消费 | 实验 17 | 已实测并讲解 |

## 依据与边界

- 当前源码：`torch/_inductor/ir.py` 的 Pointwise/Reduction/View 家族，`torch/_inductor/lowering.py` 的 `make_pointwise`、reduction lowering 与 view lowerings。
- 独立实验 05 现证明同 shape Pointwise、广播 stride-0 ReinterpretView、真实 `[2,128]` Reduction、keepdim Reduction 和 storage-backed transpose ReinterpretView。
- 综合实验 17 证明 permute/add/relu/sum/mm 的实际节点交接：前三项延迟，sum 登记 ComputedBuffer，mm 登记 ExternKernelOut；本篇只取到 sum 的计算 IR 出口。
- `[2,3]` sum 表是数学/源码推演，不是保留 Reduction 类型的运行证据；正文必须继续保留此边界。
- `transpose-layout.html` 的 24 byte 结论只适用于该 CPU contiguous float32 输入，无 padding；不推广为任意 view 的存储布局。
- 实际读者反馈尚无。

## 作者复读与迁移题

1. y 从 `[1,3]` 广播到 `[2,3]` 后，为什么 add 的 ranges 不变而两行能读同一 y？答：add ranges 由共同输出 shape 决定；y 的 ReinterpretView 在广播维使用 stride 0，使 `[0,c]` 与 `[1,c]` 落到同一槽位。
2. `sum(dim=1,keepdim=True)` 为什么 reduction_ranges 仍是 `[128]`？答：keepdim 只在输出坐标保留一个 size-1 位置；组内仍要遍历原 dim 1 的 128 项。
3. 把综合例从 `sum(dim=1)` 改成 `dim=0`，哪些索引角色交换？答：q 成为保留的输出坐标，p 成为遍历三项的 reduction index，输入仍先经 permute 映射到 `[q,p]`。
4. transpose 实验 operations=0 是否说明它复制了零字节的新 tensor？答：不是。ReinterpretView 共享原 InputBuffer，只改变 size/stride/offset 的坐标解释。
5. 若广播输入尚未有固定 storage，是否仍能断言必为 ReinterpretView？答：不能；`ExpandView.create` 可保留 ExpandView/reindexer，语义同样是清零广播维，具体表示取决于底层是否已有 storage/layout。
6. sum 后接 mm，为什么本篇在 sum 处结束而不解释 mm 算法？答：本篇出口已经给出 Reduction 的迭代与访问合同；mm 是 Q24 的特殊调用职责，只作为消费者证明 buf0 被使用。

答案包含具体 index、ranges、stride 或消费者，均可由正文推导。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K07-R 完成，Q22 达到第 2 层本篇完成条件。
- 小/空/拆分 reduction 的完整构造分支留在 Q221 叶子逐行回修，未用本篇概括代替叶子完成。
- 根队列下一项为同层 Q23；Q221 仍按第 3 层待回修。

## 第五轮重新复核

- 初次检查发现覆盖表把“小/空/特殊 reduction”列为主要情况，正文却只有“小规约可能展开”的概括，因此 Q22 未直接沿用旧完成结论。
- 已扩展实验 05，直接观察 zero-size sum 与 size-one argmax 都返回 Pointwise，zero-size max 明确抛出不支持异常；正文新增具体 `[2,0]`、`[2,1]` 状态推演和迁移题。
- Pointwise、广播、普通/keepdim/退化 Reduction、storage-backed/延迟 View 与综合 View→Pointwise→Reduction 现在都有本层入口、索引变化和出口。
- `[2,3]` 求和继续只作为数学例，`[2,128]` 才支持当前版本保留 Reduction 的事实主张；实验和理解验收分开。实际读者反馈仍为空。
