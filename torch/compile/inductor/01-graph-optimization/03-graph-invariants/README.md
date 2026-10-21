# 图改写后同步四份状态：顺序、形状记录、代码与引用合法性

[返回图优化](../README.md) · [上一篇：Pattern Matcher](../02-pattern-matching/README.md) · [返回总览](../../README.md)

实现叶子：[稳定拓扑排序完整源码](01-stable-sort-source/README.md)；[FakeTensorUpdater 完整源码](02-fake-meta-source/README.md)。

以 `a=x+y; b=relu(a)` 为例，a 的 `users` 集合记录“b 正在引用 a”。若把 b 改为读取 x，这条边和 users 集合必须一起更新。改边还可能改变 b 的结果形状，所以除了引用关系，还要同步节点顺序、形状等记录和实际执行的 Python `forward`。

编译器不必真的分配数据就能推导这些属性。它可以用一份不含真实元素、但保存 shape、dtype、device 和 stride 的张量替身执行节点；源码把这种替身称为 FakeTensor，把推导结果放在 `node.meta["val"]`。例如输入替身的 shape 是 `[2,3]`，转置节点得到的替身 shape 是 `[3,2]`、stride 是 `[1,3]`。

一次局部替换接好了 users，仍不代表整张 FX 图已经能安全交给 Lowering。图同时存在四类需要保持一致的事实：依赖顺序、上述 FakeTensor 记录、GraphModule 实际执行的 Python `forward`，以及节点所有权和调用目标等结构条件。

```text
局部 pass 修改 nodes / args / kwargs / users
             │
             ├─ stable_topological_sort：恢复依赖顺序
             ├─ FakeTensorUpdater：重算受影响的 node.meta["val"]
             ├─ GraphModule.recompile：由 Graph 重新生成 forward
             └─ graph.lint：验证结构是否合法
             │
             ▼
       可继续交给下一项图改写或 GraphLowering 的 GraphModule
```

四项不是重复保险：它们检查或修复不同对象。

## 拓扑顺序：输入必须先于消费者

FX `Graph` 既保存 node 的依赖边，也保存一条线性 node 顺序。合法拓扑要求：一个 node 的所有 `args/kwargs` 输入必须已经出现在它之前。

独立实验 [`experiments/03_graph_invariants.py`](../../experiments/03_graph_invariants.py) 人为构造：

```text
线性顺序：y, add, x
依赖关系：add(y, x)
```

此时 `graph.lint()` 报告 x 在定义前被 add 使用。`stable_topological_sort()` 修复后为：

```text
y, x, add
```

y 与 x 彼此没有依赖，算法保留原图中允许保留的相对顺序；add 必须等两者 ready 后才能放入结果。这就是“stable”的意义：只为满足依赖而移动，不任意重排无关节点。

### 当前稳定排序怎样推进

源码维护三类状态：

- `pending`：还要检查的节点；
- `ready`：已经排好且依赖可用的节点；
- `waiting[dependency]`：仍等某项输入的消费者。

处理 node 时收集尚未 ready 的输入。若有，就让该 node 等最后一项未就绪输入；否则把它接到当前 cursor 后，并唤醒等待它的节点。结束时 `waiting` 必须为空，`ready` 数量必须等于图中节点数。

若图真的有环，就无法得到覆盖全部节点的拓扑顺序；FX 的数据流图在交给这些 pass 时应是有向无环图，英文缩写为 DAG。

## FakeTensor meta：边改了，shape/stride 结论也可能变

FX node 的 `meta["val"]` 常保存 FakeTensor 或符号值。Pattern 的 `extra_check`、布局处理和 Lowering 都可能读取它的 shape、stride、dtype、device、storage/alias 信息。

假设 pass 把：

```text
z = add(x, y)       # 原先 x,y 都是 [2,3]，z.meta shape=[2,3]
```

改成使用可广播的 `y2`，其 shape 为 `[1,3]`。数值 target 仍叫 add，但参数对象已经变化；旧 meta 不能因为 target 没变就继续沿用。

`FakeTensorUpdater` 在 pass 前记录每个 node 的 hash：node 本身、target，以及 `args/kwargs` 容器身份。FX 修改参数时会安装新的不可变 args/kwargs 容器，因此 hash 会改变。`incremental_update()` 找出新增或参数/target 发生变化的可处理节点，再用 fake 参数重新调用 ATen target。这里的 `V.fake_mode` 是当前编译任务安装的 FakeTensorMode 上下文：它让 ATen 调用只计算输出的 shape、stride、dtype、device 和存储关系，不读取或分配真实元素。

若新 fake 结果与旧值在当前比较函数检查的类型、shape、layout、stride、offset、device 和 storage 关系上相同，就停止向后传播；若不同，则更新 `meta["val"]`，并把该 node 的 users 加入待处理集合。动态形状还可能产生不能从输入尺寸直接算出的符号，例如数据相关操作返回的长度记为 `u0`；当某个节点取得它的运行时标量值时，binding 会记录“`u0` 由这个值提供”。更新 fake 结果时这项对应关系也要同步。这样 updater 只沿真正受影响的下游重算，而不是每次完整运行全图。

实验用一条 `add→relu` 链观察了两种出口。第一张图最初让 add 读取两个 `[2,3]` placeholder；updater 建立初始 hash 后，实验把 add 的两个输入改接到两个 `[4,3]` placeholder：

| 时刻 | add meta shape | relu meta shape | 增量更新实际调用的 fake op |
| --- | --- | --- | --- |
| 改边前 | `[2,3]` | `[2,3]` | 尚未调用 |
| 改边后 | `[4,3]` | `[4,3]` | `aten.add.Tensor`、`aten.relu.default` |

add 的 hash 先变化；重算结果与旧 meta 不同，所以 add 更新并把唯一 user relu 加入待处理集合。relu 自己的 args 容器没改，但它因处于传播集合仍被重算，最后两层 meta 一起变为 `[4,3]`。

第二张图只把 add target 改为 sub，输入仍是两个 `[2,3]` tensor。记录器只看到 `aten.sub.Tensor`，没有看到下游 `aten.relu.default`。原因不是 updater 没发现 target 变化，而是 sub 的 fake 输出与旧 add 输出满足当前“相同 meta”判断，于是传播在当前节点停止。数值语义由新的 target 与稍后的 recompile 负责；FakeTensorUpdater 只维护它承诺的 tensor 属性，不计算真实数值。

它只处理当前实现明确支持的 `call_function` ATen overload 和 `operator.getitem`；拿不到完整 fake args 的节点会跳过。这里的“跳过”不是成功补齐 meta：调用模块、任意 Python callable 或仍含无 meta FX Node 的参数不会由这次 updater 执行。局部 replacement 会尽力复制输出 meta，阶段级 updater 再处理其支持范围内发生变化的依赖链。

## recompile：Graph 和可执行 forward 是两份关联状态

`GraphModule` 持有 FX Graph，同时也持有由它生成的 Python `forward`。修改 Graph node 并不会自动改写已经安装的 `forward`。

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

重新生成的代码明确为 `add = a * b`。变量名仍叫 add 不影响 target 语义；FX node name 是定位标签，不是执行操作的依据。

因此 pass 链若修改了 Graph 却忘记 recompile，直接调用 GraphModule 可能执行旧程序。当前 pre-grad、joint（发生变更时）、post-grad 都在对应出口重新编译。

## lint：验证而不是自动修复

`graph.lint()` 当前检查包括：

- node opcode 是否属于 FX 支持集合；
- node 是否确实属于这张 graph，并存在于查找侧表；
- 每个输入是否属于同图并已经在当前 node 前定义；
- node name 是否唯一；
- `call_function` target 是否 callable；
- `call_module/get_attr` 的 target 是否能在 owning GraphModule 中解析，并具有合适对象类型。

lint 发现问题就报错，它不会替作者决定怎样排序、怎样补 meta 或怎样改 target。所以常见顺序是先执行明确的修复/更新，再 `recompile()` 和 `lint()` 验收。

## users 与死代码仍是另一项状态

FX 在修改 `args/kwargs`、调用 `replace_all_uses_with`、`erase_node` 时维护 users。一个 node 没有 users 并不必然可以删除：有副作用的 node 仍可能必须执行。`eliminate_dead_code()` 因此还要依据 `is_impure` 判断。

这与拓扑排序不同：排序决定保留节点的合法先后；DCE 决定哪些节点可移除；users 支持两者判断数据消费关系。

实验把一个无 user 的纯 `aten.sin` 和一个无数据结果 user 的 `aten.copy_` 放在同图，输出直接引用被修改的 x：

```text
DCE 前：sin(y)      # 结果没人使用，纯计算
        copy_(x,y)  # 返回值没人使用，但会修改 x
        return x

DCE 后：copy_(x,y)
        return x
```

`eliminate_dead_code()` 先要求图已通过 lint，然后逆序检查。sin 无 users 且不 impure，所以删除；copy_ 虽无 users，但 schema 表明它有写入副作用，所以保留。若调用者提供自定义 `is_impure_node`，DCE 使用该判断。当前 FX 源码还明确警告默认副作用覆盖并不完备；不能把“没有 users”当成普遍安全的删除证明。

## 三个处理阶段的收尾差异

| 阶段 | 当前出口动作 |
| --- | --- |
| pre-grad | stable sort，quant lift，lint，recompile，可选数值检查 |
| joint | 只有实际 count 非零时 stable sort、lint、recompile |
| post-grad | 先可选 DCE；改写后 stable sort、constructor move、FakeTensorUpdater；最后 reinplace/decompose/FSDP all-gather reinplace，再 recompile、lint |

差异来自各阶段做过的变换和下一消费者要求。pre-grad 无论 matcher 是否命中都会整理顺序，并在 recompile 前 lint；joint 只在它累计的 `count` 非零时执行三项收尾；post-grad 在 pattern 和自定义 pass 后统一排序、移动 constructor、增量更新 fake meta，再把会重新引入 mutation 的操作留到末尾，最后 recompile 和 lint，随后才交给 GraphLowering。

## 用一个新改写判断需要哪些收尾

若 pass 只把常数 1 换成常数 2，没有新增 node，也未改变 shape/dtype：依赖拓扑通常不变，fake tensor 属性可能相同；但 GraphModule 的 `forward` 仍需 recompile，lint 仍可验证结构。

若 pass 新建 transpose 并把多个 users 接到它：可能改变插入顺序、shape/stride meta 和下游参数，四项收尾都相关；若旧 node 无 users，还要另外判断其纯度后做 DCE。

把这些状态按因果顺序串起来，可以判断一次更复杂的改写。假设原图有 `z=add(x,y); a=relu(z); b=sin(z)`，pass 新建 `transpose(z)` 并让 a、b 都改读它，但把 transpose 暂时插在 z 前面：

1. users/args 已表达新依赖，但线性顺序违法；stable sort 把 transpose 放到 z 后，并保证 a、b 又在 transpose 后。
2. transpose 的输出 shape/stride 与 z 不同；updater 重算 transpose，并沿两个 user 分支更新 a、b 的 meta，直到某条分支属性不再变化。
3. GraphModule 仍安装旧 forward；recompile 才让实际调用执行新边和新 target。
4. lint 验证顺序、所有权和 target 等结构，但不会替代前三步。
5. z 仍被 transpose 使用，不能 DCE；若改写遗留另一个无 users 纯节点，才按 purity 删除。

反过来，若只把 add 改成同 shape 的 sub，第一步通常无需移动节点，第二步在 sub 处停止，但第三步仍不可省略，因为实际数值操作已经改变。这正是四类维护不能合并成一个“图更新”动作的原因。

到这里，图优化这一职责的第二层已经完整覆盖：何时做 pass、如何匹配替换、怎样恢复图不变量。广度优先队列下一项进入 Lowering 的 `GraphLowering` 解释器。
