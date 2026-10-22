# 存储与数据版本：什么时候产生 buffer，谁和谁共享

[返回 Lowering](../README.md) · [上一篇：计算 IR](../02-compute-ir/README.md) · [返回总览](../../README.md)

实现叶子：[物化、布局固定与 mutation 源码](01-storage-source/README.md)。

计算 IR 可以只保存“给定位置怎样计算”的规则。本篇决定这条规则何时必须成为一项有名
结果、结果采用什么 layout、view 是否共享旧 storage，以及写入如何推进同一 storage 的
数据版本。

```text
延迟计算规则
   │
   ├─ 后继仍能组合 loader ─────────────→ 继续延迟
   │
   ├─ output / 外部调用 / 布局要求 ────→ realize 为有名 buffer
   │                                      └─ 冻结 layout
   │
   ├─ view ─────────────────────────────→ 新坐标，共享原 storage
   │
   └─ mutation ─────────────────────────→ 保留 storage 身份，推进数据版本
```

这些路径最终形成 operations、layout、alias 关系和 mutation target；有 operation 的路径进入 Scheduler，纯 view 的输出关系由 wrapper/output 生成路径使用。

## realize：登记一项必须产生的结果

对 `a=x+1; b=a*2`，a 可以先是一条延迟规则，b 继续把乘 2 接在它后面。若改成
`z=mm(a,w)`，已有矩阵乘实现需要一份真实张量参数，a 就必须有稳定名字和 layout：

```text
原来：a → Pointwise(i ↦ x[i]+1)

realize 后：a → 读取 buf0[i]
             op0 → 按原 Pointwise 规则产生 buf0
```

`StorageBox` 让已有引用在底层表示从 Pointwise 换成 `ComputedBuffer buf0` 后继续指向同一
逻辑结果；`TensorBox` 是节点之间传递 tensor 值的外层对象。realize 在编译 IR 中登记
buffer 和 operation，不在编译时执行计算，也不立即分配运行时内存。

常见触发者包括：图输出需要稳定返回 storage、外部函数需要真实 tensor 参数、消费者要求
固定 stride，以及 mutation 前必须保存仍读取旧版本的结果。

## layout：把逻辑坐标换成物理位置

layout 至少记录 size、stride 和 offset。`FlexibleLayout` 已确定 size，但仍允许后续根据
消费者要求选择 stride；一旦约束确定，它变成不可继续改动的 `FixedLayout`。

以 size `[2,3]` 为例：

| layout | stride | 逻辑 `[row,column]` 的元素位置 | 物理顺序 |
| --- | --- | --- | --- |
| 行连续 | `[3,1]` | `3×row+column` | `[0,1,2,3,4,5]` |
| 列连续 | `[1,2]` | `row+2×column` | `[0,3,1,4,2,5]` |

两种 layout 都容纳同样六个逻辑值，但后端生成的地址不同。内部怎样从实际读取式收集
stride 线索、选择循环填充顺序并冻结 layout，见
[实现叶子](01-storage-source/README.md)。父层出口只要求所有消费者随后读取同一份已确定
layout。

## view：新坐标继续读取旧 storage

连续 `[2,3]` 输入的 stride 是 `[3,1]`。`x.permute(1,0)` 产生 size `[3,2]`、stride
`[1,3]` 的 view：

```text
view[2,1] 的元素位置 = 2×1 + 1×3 = 5
                       = 原 x[1,2]
```

这条路径只增加坐标解释，不登记转置 operation。view 和 x 仍引用同一 storage；若后继
只能接受连续输入，存储职责再为它准备满足合同的新结果。

别名出口也可让一项计算直接写进已有结果的一段区域。无论具体实现是哪种，后续消费者
都要知道哪些名字共享同一 storage、哪个名字拥有分配、哪些名字只是另一套坐标解释。
具体 alias layout 分支留在实现叶子。

## mutation：storage 身份不变，数据版本改变

请求：

```python
old = x + 1
x.copy_(y)
return old, x
```

取 `x=[0,1,2]`、`y=[9,9,9]`。old 要得到 `[1,2,3]`，调用者持有的原 x storage 要更新
为 `[9,9,9]`。

Lowering 在同一 storage 上区分两个时间版本：

```text
覆盖前的 x ──→ old 的读取规则 ──→ 物化为 buf0
       │
       └─ copy_ 登记 mutation operation
              target = 原输入 x 的 storage
              source = y
```

登记 mutation 时，Lowering 先让仍可能读取旧 x 的结果拥有稳定数据，再登记写入原输入
storage 的 operation。它交出的编译状态为：

| 状态 | 内容 |
| --- | --- |
| `operations` | 产生 old 的 op0；写入 x 的 op1 |
| `mutated_buffers` | 原输入名 `arg0_1` |
| mutation target | op1 指向调用者持有的 x storage |
| `graph_outputs` | 普通输出 old；x 的变化由输入副作用表达 |

这正是本层边界：Lowering 已声明谁读旧版本、谁写哪个真实 storage。Scheduler 下一步把
这些登记变成执行顺序；Codegen 和 wrapper 再按顺序发出实际写入。本篇不提前决定它们
如何分组、何时提交或怎样重建用户可见返回 alias。

## 三条路径分别交给谁

| 请求 | 存储决定 | operations | 后续消费者 |
| --- | --- | --- | --- |
| `return (x+1)*2` | 组合到最终结果后才 realize | 一项产生输出的 operation | 一项读 x、写输出的工作及最终 layout |
| `return x.permute(1,0)` | 只建立共享 storage 的 view | 无转置 operation | Scheduler 没有新节点；wrapper/output 路径使用 graph output 的 alias/layout |
| `old=x+1; x.copy_(y)` | old 固定为 buffer；写回指向原 x | 旧值结果 operation + mutation operation | 旧读、写入目标、共享 storage 身份和版本关系 |

普通物化交出有名 buffer 和固定 layout；view 交出共享关系；mutation 额外交出旧读保护与
写入目标。下一篇[特殊调用](../04-special-calls/README.md)会使用这些结果作为已有函数、
fallback 和控制流节点的参数。
