# 图状态维护：修复三类派生状态，再验证 Graph

[返回图优化](../README.md) · [上一篇：Pattern Matcher](../02-pattern-matching/README.md) · [返回总览](../../README.md)

实现叶子：[稳定拓扑排序](01-stable-sort-source/README.md)；[FakeTensor 属性更新](02-fake-meta-source/README.md)。

Pattern Matcher 交来的是已经改变的节点、参数引用和 users。这些节点关系是当前改写的
主体；围绕它们还有三类可能过期的派生状态，最后还要验证 Graph 自身满足结构约束：

```text
已经修改的 nodes / args / kwargs / users
             │
             ├─ 修复节点顺序：生产者排在消费者之前
             ├─ 刷新 node.meta：结果属性尽量对应新的计算关系
             ├─ 重建 GraphModule.forward：执行新的 Graph
             └─ 验证结构约束：所属图、名称、target 和引用合法
             │
             ▼
       内部记录一致的 GraphModule
```

前三项会改变一份派生状态；`lint()` 不维护第四份独立记录，而是读取当前 Graph 并验证一组
约束。节点顺序本身也属于它检查的结构之一，因此不能把这四项理解成四份彼此独立的缓存。

## 谁在什么时候调用这些工具

Inductor 没有一个“每次改图后固定依次调用四个函数”的统一入口。包围一组 pass 的阶段
函数，或者某个自行改图的 pass，负责选择需要的维护动作。当前三条阶段主路径就不同：

| 阶段入口 | 当前源码中的收尾路径 |
| --- | --- |
| pre-grad | 一组改图结束后稳定拓扑排序；随后还有 `quant_lift_up`；再 `lint()`、`recompile()` |
| joint | 只有本轮确实发生变化时，稳定拓扑排序、`lint()`、`recompile()` |
| post-grad | 改图前先创建 `FakeTensorUpdater` 记录旧节点状态；改图后稳定拓扑排序、移动 constructor、增量更新 fake meta；随后还有 reinplace 等 pass；最后 `recompile()`、`lint()` |

因此，本篇讲的是四种不同责任及其数据关系，不是一个跨阶段都相同的调用协议。排序、meta
更新和 recompile 是否需要、放在哪里，由拥有这轮改图范围的调用者决定；`lint()` 只能
验证调用当时的 Graph 结构，不能证明 meta 或生成的 `forward` 已刷新。

## 节点顺序必须满足依赖

FX `Graph` 同时保存依赖边和一条线性节点顺序。若 `add` 读取 x，那么 x 必须出现在
`add` 前面。

独立实验 [`experiments/03_graph_invariants.py`](../../experiments/03_graph_invariants.py)
构造了下面的坏顺序：

```text
线性顺序：y, add, x
依赖关系：add(y, x)
```

`graph.lint()` 报告 x 在定义前被 add 使用。稳定拓扑排序把顺序修成：

```text
y, x, add
```

y 和 x 彼此没有依赖，因此保留原先相对顺序；add 必须等两项输入都已出现。若依赖形成
环，就不存在覆盖全部节点的合法拓扑顺序。排序内部怎样维护等待项和可放置项，见
[稳定拓扑排序实现](01-stable-sort-source/README.md)。

## node.meta 必须对应新的输入和结果

`node.meta["val"]` 常保存 FakeTensor 或符号值。FakeTensor 没有真实元素，但记录 shape、
dtype、device、stride 和存储关系。Pattern 条件、Lowering 和布局处理都会读取这些属性。
`FakeTensorUpdater` 的目标是把其中能够安全重算的部分刷新，而不是保证任意改写后所有 meta
字段必然正确。

实验先建立 `add→relu`，两项输出的 shape 都是 `[2,3]`；然后把 add 改接到两项
`[4,3]` 输入：

| 时刻 | add meta shape | relu meta shape | 重新执行的 fake op |
| --- | --- | --- | --- |
| 改边前 | `[2,3]` | `[2,3]` | 无 |
| 改边后 | `[4,3]` | `[4,3]` | `aten.add.Tensor`、`aten.relu.default` |

add 的输入关系改变后，属性更新先重算 add；结果属性变化，再沿 users 重算 relu。第二个
对照只把 add target 改成同 shape 的 sub。sub 的结果属性仍与旧 add 相同，属性传播在
sub 停止；实际数值操作则由新的 target 和稍后的 `recompile()` 体现。

当前 updater 只执行它明确支持、并且拥有完整 fake 参数的节点。内部怎样发现变化、比较
新旧属性以及推进下游工作，见
[FakeTensor 属性更新实现](02-fake-meta-source/README.md)。

当前比较逻辑还有一个必须记住的边界：它比较 shape、layout、stride、storage offset、
device 和 storage 关系，但没有比较 dtype。实验把 `aten.add.Tensor` 改成
`aten.eq.Tensor` 时虽然重新执行了 eq，节点的旧 `val` 仍保持 float32，没有更新成 bool。
缺少输入 fake 值或 target 不是当前支持的 ATen/getitem 时也会跳过。遇到这些情况，不能用
“调用过 updater”推断 meta 已完全可靠；pass 要避免这种改写、自己维护 meta，或使用更完整
的 fake propagation。

## recompile 让 forward 执行新的 Graph

`GraphModule` 持有 FX Graph，也持有由它生成的 Python `forward`。修改节点不会自动改写
已经安装的 `forward`。

实验最初 trace：

```python
def fn(a, b):
    return a + b
```

随后只把 call node 的 target 改成 `operator.mul`。输入 2、3 时：

| 时刻 | Graph target | 实际 `gm(a,b)` |
| --- | --- | --- |
| 修改后、recompile 前 | mul | 5，旧 forward 仍做加法 |
| `gm.recompile()` 后 | mul | 6，新 forward 做乘法 |

重新生成的代码虽然仍可沿用旧变量名，执行操作已经由节点 target 决定。`recompile()` 的
出口是一份与当前 Graph 对应的可执行 `forward`。

## lint 验证整张图的结构

`graph.lint()` 检查：

- opcode 是否属于 FX 支持集合；
- 节点是否属于当前 graph，名称是否唯一；
- 输入是否属于同一张图并在使用前定义；
- `call_function` target 是否可调用；
- `call_module` 和 `get_attr` 的 target 是否能在 owning GraphModule 中解析。

`lint()` 发现问题就报错。它不执行排序、不重算 meta，也不重新生成 `forward`。有些阶段
在 `recompile()` 前 lint，有些阶段在其后 lint；这不改变 lint 的职责边界。

## 用同一个教学改写区分四项责任

原图返回一个二元 tuple，节点 target 分别是 `aten.add.Tensor`、`aten.relu.default` 和
`aten.sin.default`：

```text
x,y → aten.add.Tensor(z) ─┬→ aten.relu.default(a) ─┐
                           └→ aten.sin.default(b)  ─┴→ output((a,b))
```

独立实验 [`experiments/03_graph_invariants.py`](../../experiments/03_graph_invariants.py) 给出
一条确定的组合路径。它先对原图创建 `FakeTensorUpdater`，让 updater 记住改图前的节点；
随后在 z 前创建 `t = aten.transpose.int(z, 0, 1)`，并让 relu、sin 都从读取 z 改为读取 t。
x、y 的 shape 都是 `[2,3]`。创建 t 时没有复制 meta，所以 `"val" in t.meta` 为 false；
relu 和 sin 仍保存改边前的 `[2,3]` fake 值。此时线性顺序实际是：

```text
x, y, t, z, relu, sin, output
```

实验驱动代码随后依次调用稳定拓扑排序、`incremental_update()`、`recompile()` 和 `lint()`。
这条顺序用于把一个案例走完整，并与 post-grad 的相关数据关系一致；它仍不表示所有产品阶段
都使用相同流水。

| 状态 | 修改刚发生时 | 维护后的结果 |
| --- | --- | --- |
| 节点顺序 | `x,y,t,z,relu,sin,output`；t 排在它读取的 z 前 | `x,y,z,t,relu,sin,output`，每项输入先出现 |
| node meta | t 没有 `val`；relu、sin 仍为 `[2,3]` | updater 执行 transpose，并重算两个 users；t、relu、sin 的 shape 为 `[3,2]`，stride 都是 `[1,3]`；z 保持 shape `[2,3]`、stride `[3,1]` |
| `forward` | 仍执行 `relu(z)`、`sin(z)` | 生成的代码改为先求 t，再执行 `relu(t)`、`sin(t)` |
| 图结构 | 尚未统一验证 | `lint()` 确认引用、所属图、名称和 target 合法 |

若负责这轮改写的调用者按需要完成这些维护，GraphModule 保存的新关系是：

```text
x,y → add(z) → transpose(t) ─┬→ relu(a)
                              └→ sin(b)
```

这张 GraphModule 可以交给下一项 pass；若图优化已经结束，就交给 GraphLowering。旧节点
是否应删除由执行替换的规则或相应阶段决定，本篇维护的是最终留下来的图。

最终节点接口可以完整列成：

| 名称 | FX `op` | `target` | `args` | shape | stride |
| --- | --- | --- | --- | --- | --- |
| x | `placeholder` | `"x"` | `()` | `[2,3]` | `[3,1]` |
| y | `placeholder` | `"y"` | `()` | `[2,3]` | `[3,1]` |
| z | `call_function` | `aten.add.Tensor` | `(x,y)` | `[2,3]` | `[3,1]` |
| t | `call_function` | `aten.transpose.int` | `(z,0,1)` | `[3,2]` | `[1,3]` |
| a | `call_function` | `aten.relu.default` | `(t,)` | `[3,2]` | `[1,3]` |
| b | `call_function` | `aten.sin.default` | `(t,)` | `[3,2]` | `[1,3]` |
| output | `output` | `"output"` | `((a,b),)` | 二元 tuple | — |

实验重新生成的 `forward` 去掉释放临时变量的语句后，计算与返回结构是：

```python
def forward(self, x, y):
    z = torch.ops.aten.add.Tensor(x, y)
    t = torch.ops.aten.transpose.int(z, 0, 1)
    a = torch.ops.aten.relu.default(t)
    b = torch.ops.aten.sin.default(t)
    return (a, b)
```

这段代码整理了实验生成代码中的临时变量名，但 target、参数和 tuple 返回结构保持一致。

本实验中“谁调用”是实验驱动代码；产品中对应责任属于上文列出的阶段入口或自行改图的
pass。实验记录的 fake 执行包括 `aten.transpose.int`、relu、sin，以及 fake machinery 内部
读取 device 的 `prim.device.default`；关键因果是新 t 先被重算，属性变化再使两个 users
获得 `[3,2]`。

把对照条件改成“只将 add 换成同 shape 的 sub”时，节点顺序不变，meta updater 重新执行
sub 后发现所比较的属性不变，不再推进到 relu；`recompile()` 仍要让实际调用执行减法，
`lint()` 仍可验证图结构。由此可以分别判断三类派生状态和结构验证是否需要动作，而不把
它们压成一个含义不清的“更新图”。
