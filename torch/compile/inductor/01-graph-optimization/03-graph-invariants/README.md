# 图状态维护：让四份程序记录重新一致

[返回图优化](../README.md) · [上一篇：Pattern Matcher](../02-pattern-matching/README.md) · [返回总览](../../README.md)

实现叶子：[稳定拓扑排序](01-stable-sort-source/README.md)；[FakeTensor 属性更新](02-fake-meta-source/README.md)。

Pattern Matcher 交来的是已经改变的节点、参数引用和 users。要让下一项 pass 或
GraphLowering 安全读取这张图，还要同步四份状态：

```text
已经修改的 nodes / args / kwargs / users
             │
             ├─ 节点顺序：生产者排在消费者之前
             ├─ node.meta：结果属性对应新的计算关系
             ├─ GraphModule.forward：执行新的 Graph
             └─ 图结构：节点所属图、名称、target 和引用合法
             │
             ▼
       内部记录一致的 GraphModule
```

四项分别维护节点的线性排列、编译期结果属性、可执行 Python 代码和结构约束。

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

`lint()` 发现问题就报错。它不执行排序、不重算 meta，也不重新生成 `forward`；前三项先
把各自记录更新好，`lint()` 再验证留下来的图。

## 同一个改写怎样经过四项维护

原图为：

```text
x,y → add(z) ─┬→ relu(a)
               └→ sin(b)
```

一个 pass 新建 `transpose(t)`，让 relu 和 sin 都改读 t；新节点暂时插在 z 前面。假设
x、y 的 shape 是 `[2,3]`，transpose 交换两维。

| 状态 | 修改刚发生时 | 维护后的结果 |
| --- | --- | --- |
| 节点顺序 | t 排在它读取的 z 前 | `x,y,z,t,a,b`，每项输入先出现 |
| node meta | t、a、b 仍可能保留旧 `[2,3]` 记录 | t、a、b 的 shape 为 `[3,2]`；连续 `[2,3]` 输入的 t stride 为 `[1,3]` |
| `forward` | 仍执行 `relu(z)`、`sin(z)` | 生成的代码改为先求 t，再执行 `relu(t)`、`sin(t)` |
| 图结构 | 尚未统一验证 | `lint()` 确认引用、所属图、名称和 target 合法 |

四项完成后，GraphModule 保存的新关系是：

```text
x,y → add(z) → transpose(t) ─┬→ relu(a)
                              └→ sin(b)
```

这张 GraphModule 可以交给下一项 pass；若图优化已经结束，就交给 GraphLowering。旧节点
是否应删除由执行替换的规则或相应阶段决定，本篇维护的是最终留下来的图。

把对照条件改成“只将 add 换成同 shape 的 sub”时，节点顺序不变，meta 传播在 sub
停止，但 `recompile()` 仍要让实际调用执行减法，`lint()` 仍验证图结构。由此可以分别
判断四项状态是否需要改变，而不把它们压成一个含义不清的“更新图”。
