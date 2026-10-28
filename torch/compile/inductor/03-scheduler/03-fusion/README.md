# 融合：哪些调度节点可以合成一个生成单元

[上一篇：排序与生存期](../02-order-lifetime/README.md) · [下一篇：后端分派与生成顺序](../04-backend-dispatch/README.md) · [返回 Scheduler](../README.md) · [返回总览](../../README.md)

Q32 交来一列仍然存活的 scheduler node。每个节点已经记录了读写位置、尚未由前面结果
满足的依赖、使用它的节点、全部祖先，以及它在当前顺序中的位置。融合要从中挑出可以放进
同一个 kernel 或同一个外层循环的工作。它不再修改 FX 图，也不执行计算；它改变的是
Scheduler 随后交给生成后端的工作分组。

先看最短的例子：

```text
op0：逐位置计算 buf0[i] = x[i] + 1
op1：逐位置计算 out[i]  = buf0[i] * 2
```

分开生成时，op0 要写出完整的 `buf0`，op1 再读回来。若把两者融合，同一个循环位置可以
先算 `t=x[i]+1`，随即算 `out[i]=t*2`。融合前有两个 scheduler node，融合后是一个
`FusedSchedulerNode`，其中仍保存 op0、op1 两个成员。

这项工作按下面四步反复进行：

```text
一列 scheduler nodes
  │
  ├─ 1. 找候选：哪些节点共同使用同一份数据
  ├─ 2. 判定：依赖方向、访问位置和目标后端是否允许合并
  ├─ 3. 排序、重查并判断收益：本轮先试谁，当前组还能否合并，是否值得合并
  └─ 4. 合并：建立携带新依赖状态的 FusedSchedulerNode
         │
         └─ 节点数减少则开始下一轮；不再减少就结束
```

四步读取的入口信息如下：

| 节点信息 | 由谁建立 | 融合时的用途 |
| --- | --- | --- |
| `read_writes` | Q31 分析循环体的读取和写入 | 比较两边是否访问同名 buffer 的同一位置 |
| `unmet_dependencies` | Q31 去掉函数入口已经可用的输入和常量后留下 | 判断消费者的读取能否由生产者在融合体内满足；普通图内前驱仍作为依赖保留 |
| `ancestors` | Q32 沿拓扑顺序累计 | 判断两节点有无生产—消费关系，并防止形成环 |
| `users`、`used_buffer_names()` | Q31 建立使用关系和读写集合 | `users` 让读者看到谁消费结果；本篇找候选实际读取 `used_buffer_names()`，其中既包括读也包括写 |
| `group` | 节点建立时按后端循环范围归组 | 让 CPU、SIMD 等后端判断循环或规约形态是否兼容 |
| `min_order/max_order` | Q32 记录融合前位置；融合组取成员范围 | 避免把原来相距很远的工作绑在一起 |

## 第一步：从共同使用的数据找到候选

`get_possible_fusions()` 先按 `used_buffer_names()` 把节点分组，在每组内枚举节点对，并用
`seen` 去重。以开头的例子为例，op0 写 `buf0`，op1 读 `buf0`，所以两者都出现在
`buf0` 这一组，形成候选 `(op0, op1)`。

这里的“候选”只表示值得继续检查。两个节点共同出现了名字 `buf0`，并不等于它们能在
同一个循环位置交接数据。下面这个变体仍然共同使用 `buf0`：

```text
op0 写：buf0[i]
op1 读：buf0[(i-1) mod 8]
```

但是 op1 在位置 0 需要 op0 在位置 7 产生的值，不能把完整中间结果省掉。访问位置要到
第二步才比较。

若启用 `aggressive_fusion`，系统还会把 `group` 相同的非规约节点放在一起尝试，即使它们
没有共同使用的数据。默认配置不这样做，因为既无生产—消费关系又无共同读取时，通常看不出
融合能省下哪次内存访问。

两种复合节点使候选具有方向：

- template 节点保存由模板产生的主体计算；它生成结果后，可以把紧接着使用该结果的普通
  逐点计算接成 epilogue。因此候选中的 template 是生产者，epilogue 是消费者。
- foreach 节点内部保存一列可以成组生成的子节点。两个 foreach 融合时按列表位置配对子
  节点；普通节点与 foreach 融合时，要先找到真正读写相连的那个子节点。

候选搜索通常按原节点对先试 `(node1,node2)`；若失败，而 node2 是 template 或 foreach
节点，才反过来试 `(node2,node1)`，让后续检查拿到上述方向。

## 第二步：证明这两个节点能够一起生成

`can_fuse(node1,node2)` 先处理所有节点共有的限制，再根据祖先关系分成纵向融合和横向
融合。它回答的是“当前表示和目标后端能否正确生成这组工作”，不是预测运行时间。

### 纵向融合：消费者读取生产者的结果

若 node1 的 operation 名出现在 node2 的 `ancestors` 中，node2 依赖 node1。开头例子
就是纵向融合：

```text
op0.read_writes.writes = {MemoryDep(buf0, index=i, size=8)}
op1.unmet_dependencies = {MemoryDep(buf0, index=i, size=8)}
```

`can_fuse_vertical()` 比较生产者的写与消费者尚未满足的读。名字、索引和范围匹配后，
op1 对 `buf0[i]` 的读取可以在融合体内由 op0 满足。其他未满足依赖仍可从融合体外部读；
但如果它们来自夹在 op0 与 op1 之间、又依赖 op0 的节点，合并会跨过这个中间节点，因而
拒绝。

本例随后通过 CPU backend 的循环兼容检查，成为一对静态允许的候选：

```text
(op0, op1)
```

这时还没有创建融合节点。它仍要经过第三步的排序、当前组重查、成环检查和收益判断；真正
合并发生在第四步。

把消费者改成滚动读取后，两个 `MemoryDep` 的索引不相等。当前 CPU 实验在更早的共享访问
收益检查中得到 `no shared data`，最终仍保留 `[op0,op1]`。即使该检查放行，纵向匹配也
不会把 `buf0[i]` 的写当成 `buf0[(i-1) mod 8]` 的读来满足。

mutation 带来一种额外依赖。Q31 为“旧读取必须先于覆盖写入”建立 `WeakDep`。这种依赖只在
融合体确实对被修改的 buffer 做同位置原地写时才能在内部满足：写入必须是一个明确的
`MemoryDep`，生产者对原 storage 的相关读取必须与该写入索引和范围相同。不同位置、间接
索引或无法确认唯一写入时，WeakDep 仍留在融合体外，不能据此做纵向融合。

alias 走的是另一条路径。普通 pointwise 通过 view 读取数据时，view 的索引变换通常已经
折进 `MemoryDep`；融合看到的是实际 buffer 名、变换后的索引和循环 `group`。若生产者的写
与消费者的读仍可精确匹配，并且 backend 接受循环形态，这对普通节点可以继续融合。若
alias 由 FallbackKernel、ComplexView 或 MultiOutput 这类 extern operation 表达，节点会因
extern 没有可内联循环体而停在 extern 边界。template 的 epilogue 则有明确门禁：consumer
报告 alias 或 mutation 时直接拒绝。由此，普通 view 的决定依据是变换后的依赖和循环，
template epilogue 读取 alias 标志，extern alias operation 读取节点种类。

### 横向融合：两项工作互不依赖

若两个节点互不为祖先，但都读取 `x[i]`，它们可以共享循环和对 x 的读取。例如：

```python
left = x + 1
right = x * 2
return left, right
```

实验在两条分支中插入测试算子 `realize`，要求前面的中间结果先成为具名 buffer；这样
Scheduler 中会保留可观察的生产—消费节点：

| 节点 | 读取 | 写入 | 祖先 |
| --- | --- | --- | --- |
| op0 | `x[i]` | `left_tmp[i]` | 空 |
| op1 | `left_tmp[i]` | `left[i]` | `{op0}` |
| op2 | `x[i]` | `right_tmp[i]` | 空 |
| op3 | `right_tmp[i]` | `right[i]` | `{op2}` |

op0→op1、op2→op3 是纵向候选；op0 与 op2 共读 `x[i]`，是横向候选。横向融合还会拒绝
原顺序距离超过 64 的节点，以免把一份中间结果的生存期拉得很长。CPU backend 要求循环
范围兼容，禁止横向融合 template，并用 `cpp.max_horizontal_fusion_size` 限制成员数。

### 不同节点种类还要通过各自的边界

下面的表描述 `can_fuse` 接到特殊情况时实际检查什么。通过通用检查后，最后一列的后端
条件仍必须成立。

| 情况 | 融合条件或拒绝原因 | 后端还要确认什么 |
| --- | --- | --- |
| 普通 pointwise | 同设备、存在精确共享访问，且成员总数不超过 `max_fusion_size` | CPU 循环范围兼容；SIMD 的 `numel/rnumel` 与 tiling 兼容 |
| reduction | 不能借 `aggressive_fusion` 越过“无共享数据”；生产—消费方向仍须正确 | CPU 只接受它支持的普通或 outer-loop 组合；SIMD 比较规约范围和 tiling |
| template + epilogue | template 必须在前；epilogue 不能含 alias/mutation，不能是 reduction，且 `epilogue_fusion` 已启用 | CPU 只接非规约 epilogue；SIMD 只接受它支持的 Triton template 和循环形态 |
| foreach + foreach | 两边成员数相同，并且对应位置的每一对成员都能融合 | 由 foreach 所在后端继续检查成员 |
| 普通 producer + foreach consumer | producer 不能是 reduction；它必须只命中 foreach 中实际读取其结果的成员 | 对该成员执行普通融合检查 |
| foreach producer + 普通 consumer | consumer 不能是 reduction；consumer 必须只依赖 foreach 中唯一一个 producer 成员 | 对该成员执行普通融合检查 |
| 普通 extern 或 nop | 没有可并入融合循环的普通循环体，拒绝 | 不再进入后端融合 |
| 不同 device、显式 `no_fuse`、Grouped node | 直接拒绝 | 不再进入后端融合 |

例如 `sin(mm(x,w))` 在当前 CPU 实验中由 extern mm 节点和普通 sin 节点组成。mm 不是可内联
的普通循环体，因此返回 `node1 is extern or nop`，两个节点保留。template 虽然也可能表现
为特殊生成节点，但它有上表所列的专用 epilogue 路径，不能与普通 extern 混为一类。

## 第三步：候选排序后，按当前融合组重新检查

同一个节点可能出现在多个候选中，Scheduler 必须决定先尝试哪一对。它先询问 backend 的
`get_fusion_pair_priority()`，只保留整数 priority 最小的一组。CPU 的完整循环融合通常为
0；只合并共同外层循环的 outer-loop fusion 为 1，所以同一轮先处理前者。较低优先级的
合法候选本轮不会尝试；只有本轮确实发生合并、进入下一轮后，它才可能以新的候选出现。

同一 priority 内再按下面的四元组从大到小排序：

```text
(
  是否符合 template/epilogue 偏好且有共享访问,
  两边是否同为规约或同为非规约且有共享访问,
  精确公共依赖估计的字节数,
  -原顺序距离,
)
```

对开头 8 个 float32 元素的同位置读写，第三项是 `8×4=32`。这是依据依赖范围和 dtype
估计的可省内存访问量，不是运行时测得的流量。相邻节点的最后一项为 -1，比相距两位的
-2 靠前。

在开头的双节点例子中，本轮只有 `(op0,op1)` 这一对：backend priority 为 0，score 为
`(True,True,32,-1)`。它映射回仍未合并的 op0、op1，第二次 `can_fuse` 仍通过，也不会形成
环。实验关闭了 benchmark，因此收益门直接允许。到这里，这一对才可以交给第四步合并。

横向案例第一轮的主要候选是：

| 候选 | 共享访问估计 | 顺序距离 | 关系 |
| --- | ---: | ---: | --- |
| op0 + op1 | 32 字节 | 1 | 左分支纵向 |
| op2 + op3 | 32 字节 | 1 | 右分支纵向 |
| op0 + op2 | 32 字节 | 2 | 两分支横向 |

处理第三对时，op0 和 op2 已经不再是两个独立组。Scheduler 用原成员名查询
`name_to_fused_node`，得到当前的 `op0_op1` 和 `op2_op3`，然后重新执行 `can_fuse`。
它还用当前组的祖先关系检查这次合并是否会形成环。最后实际发生的是：

| 原候选 | 执行时的当前组 | 结果 |
| --- | --- | --- |
| op0 + op1 | op0、op1 | op0_op1 |
| op2 + op3 | op2、op3 | op2_op3 |
| op0 + op2 | op0_op1、op2_op3 | op0_op1_op2_op3 |

因此候选生成时通过检查，并不保证执行时仍能合并。前面的合并可能扩大成员数、改变祖先
集合、触发后端限制或形成环；第二次检查使用的是这些新状态。

静态检查和成环检查通过后，`speedup_by_fusion()` 决定是否还要实测收益：

| 当前情况 | 收益判断 |
| --- | --- |
| `benchmark_fusion=False`，且不是 MultiTemplate | 直接允许 |
| CPU、foreach、非 Triton template 或含 atomic-add | 当前实现不做这项 benchmark，直接允许 |
| 普通受支持的 GPU 候选 | 分别测 node1、node2 和融合体；任一耗时为无穷大则拒绝，融合耗时严格小于两者之和才接受 |
| benchmark 抛出普通编译错误 | 错误继续抛出；特定 `Loop-carried variable` 编译错误按当前兼容处理允许融合 |
| MultiTemplate + epilogue | 即使关闭普通 benchmark 也走专用比较，见下文 |

MultiTemplate 同时保存多个实现及其未融合耗时。系统先取得最快的未融合实现耗时 `ms1`，
再测单独 epilogue 的 `ms2`；然后只尝试 Triton template 候选。某个候选未融合时已经不快于
`ms1+ms2`，后续更慢的候选就不再尝试；尝试数量还受配置限制。只有最佳融合耗时严格小于
`ms1+ms2` 时，系统才把 MultiTemplate 固定为获胜的 Triton 实现并接受融合。否则模板仍不
与该 epilogue 合并。这里要求延迟测量已经产生至少一个有效耗时：timing 表为空时，系统在
取得最快未融合实现时就失败；若所有选择的耗时都不是有限值，延迟测量会更早报告
`NoValidChoicesError`。这两种情况都是编译失败，不是“没有模板获胜，所以只拒绝融合”。

实验 10 关闭了 `benchmark_fusion` 并只运行 CPU，所以它证明候选排序、当前组映射和静态
拒绝结果，不证明 GPU 或 MultiTemplate 的耗时分支。

## 第四步：合并后建立一份可继续调度的节点状态

backend 接受一对节点后，普通路径把两个节点的成员展平，建立新的 `FusedSchedulerNode`。
它不只是保存 `members=[...]`；下一对候选和下一轮调度还要直接读取这个新节点的状态。
开头案例此时才得到 `op0_op1: FusedSchedulerNode(members=[op0,op1])`。对
`x=[0,1,2,3,4,5,6,7]`，它在每个 i 上先加 1、再乘 2，结果是
`[2,4,6,8,10,12,14,16]`。

开头例子合并前后如下：

| 状态 | op0、op1 合并前 | op0_op1 合并后 |
| --- | --- | --- |
| 成员 | 各自一个成员 | `[op0,op1]` |
| `read_writes` | op0 写 buf0；op1 读 buf0、写 out | 合并两边读写；融合体内部的 buf0 写→读不再作为外部读取 |
| `unmet_dependencies` | op1 尚需 buf0 和其他外部输入 | 去掉组内产生的 buffer 和组内写；只保留融合体外部必须先提供的依赖 |
| `ancestors` | 各自保存原祖先 | 取成员祖先并集；检查新环时再从合并祖先中扣掉本组成员名 |
| `min_order/max_order` | 各自一个位置 | 取所有成员覆盖位置的最小值和最大值 |
| `group` | 各自保存后端循环范围 | backend 已证明兼容后，有规约时取第一个规约成员的 group，否则取第一个成员的 group |
| 名字映射 | op0→op0，op1→op1 | op0→op0_op1，op1→op0_op1 |

这份状态使新节点能够立即参加后面的 `can_fuse`、成环检查和拓扑排序。每轮末，Scheduler
按 `min_order` 初排，再做一次拓扑排序并删除已经多余的 weak dependency。若本轮节点数
减少，就用结果开始下一轮；节点数不变或已经只剩一个节点时结束，最多运行十轮。

表中是普通 `FusedSchedulerNode` 的路径。CPU 遇到兼容但写法不同的循环范围时，可能先
重算一个成员的循环 group；只合并共同外层循环时会建立 `OuterLoopFusedSchedulerNode`；
foreach 也有自己的融合节点。无论具体类型是哪一种，新节点的 group 都会被下一轮候选
分组、backend 合法性检查和 priority 判断继续读取。

因此三种没有融合的情况都从同一个出口离开：输入本来只有一个节点、没有产生候选，或所有
候选在检查/收益判断中被拒绝。本轮节点数没有减少，融合循环停止，原节点列表继续进入后续
Scheduler 整理。

本篇实验还验证了两个明确拒绝出口：把 `max_fusion_size` 设为 1 时，两个单成员节点合并会
超过上限，仍保留两节点；滚动索引案例和 extern 案例也都保留原节点。数值结果仍与 eager
一致，因为拒绝融合只改变生成分组，不改变计算含义。

## 本篇出口

融合结束后得到一列普通节点和融合节点。每个融合节点已经具备继续排序和判断所需的成员、
读写、外部未满足依赖、祖先与 order 范围。Scheduler 随后还会处理循环合并、MultiTemplate
定型、通信重排和 combo/grouped node，并回到 Q32 的最终生存期步骤计算 `last_usage`。
完成这些整理后，Q34 才接收最终节点列，按 device 和节点种类交给生成后端。

读到这里，可以用下面的顺序判断一对新节点：

1. 它们因哪个共同 buffer 或相同 group 成为候选？
2. 它们是生产—消费关系，还是互不依赖的横向关系？
3. 精确访问位置、mutation 顺序、特殊节点方向和后端循环形态是否允许合并？
4. priority 和 score 会让它们在本轮何时被尝试？
5. 映射到当前融合组后，成员数、祖先和收益判断是否仍然通过？
6. 合并后的读写、外部依赖、祖先和 order 范围是什么？

这六问把“看起来可以一起算”落实为 Scheduler 实际产生的融合节点。

## 继续阅读实现

[融合主循环源码：候选、合法性、代价与循环状态](01-fusion-source/README.md)
