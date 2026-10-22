# 三个 pass 入口分别接住哪一张图

[返回图优化](../README.md) · [返回总览](../../README.md)

`torch.compile` 中，Dynamo 先把一次 Python 执行记录成 FX 图。图中的一项操作是一个节点；一个节点读取另一节点的结果，就形成一条引用。`GraphModule` 保存这张图，并提供按照节点关系执行的 `forward`。

图优化不会只运行一次。Dynamo 之后还有一段 AOTAutograd 流程：它会整理算子写法和原地操作，训练时还要生成求梯度的计算。Inductor 在这段流程的前后设置了三个入口：pre-grad、joint 和 post-grad。每个入口都会调用一组 pass。这里的 pass 是“一轮检查图、在条件满足时修改节点和引用的程序”。入口负责按顺序运行这些 pass；具体 pass 才决定要找什么节点、怎样改图。

先从第一个入口开始，因为它直接接住 Dynamo 交出的图。

## pre-grad 怎样完成一次改图

`pre_grad_passes` 接收 Dynamo 产生的 FX `GraphModule`，按当前配置依次调用 pre-grad pass。某条 pass 找到合适的节点，图就改变；所有 pass 都没有找到目标，计算关系便保持原样。入口结束时会把新节点排在使用者之前，并检查节点引用是否合法。`GraphModule.forward` 是根据图生成的 Python 代码，改节点不会自动改好这段代码，因此入口还要重新生成它。整理后的 `GraphModule` 随后交给 AOTAutograd。

下面用当前默认启用的 `batch_relu` 走完一次实际改写。假设图里有五个互不依赖的 ReLU：

```text
a0(shape=[2, 3]) ──> relu0 ──> use0
a1(shape=[2, 3]) ──> relu1 ──> use1
a2(shape=[2, 3]) ──> relu2 ──> use2
a3(shape=[2, 3]) ──> relu3 ──> use3
a4(shape=[2, 3]) ──> relu4 ──> use4
```

五个节点调用的都是 `torch.nn.functional.relu(..., inplace=False)`。ReLU 逐元素把负数变成 0，其他数保持不变。把五个 `[2, 3]` Tensor 叠成 `[5, 2, 3]` 后做一次 ReLU，再沿第 0 维拆开，五份结果分别与原来的 `relu0` 至 `relu4` 相同。

规则按输入 shape 和 `inplace` 参数分组。shape 相同才能叠成规则预期的批量输入，`inplace=False` 保证旧节点不会修改各自的输入。五条路径互不读取彼此的结果，所以合并不会引入原图没有的先后依赖。当前配置把收益门槛设为五项；这是决定是否值得合并的配置，不是保持数值相同所需的第五项语义条件。少于五项时，这条规则不改图。

达到门槛后，pass 修改节点和引用：

```text
改写前：
a0 ─> relu0 ─> use0
a1 ─> relu1 ─> use1
...                    共五条

改写后：
a0 ─┐
a1 ─┤
... ├─> stack [5,2,3] ─> relu [5,2,3] ─> unbind ─> item0 [2,3] ─> use0
a4 ─┘                                               ├> item1 [2,3] ─> use1
                                                    └> ...          ─> use4
```

具体有四步：

1. 新建 `stack([a0, a1, a2, a3, a4], dim=0)`；
2. 新建一个 ReLU，让它读取 stack 的结果；
3. 新建 `unbind`。它产生包含五项结果的序列；五个 `getitem` 分别从序列中取得一项；
4. 让每个 `useN` 改读对应的 `itemN`，然后删除五个已经无人使用的旧 ReLU 节点。

寻找候选节点、检查 shape、`inplace` 和数量条件的部分称为 matcher；命中后执行上面四步的代码负责 replacement。pre-grad 入口依次运行包含这两部分的 pass，最后完成图的整理和检查。

这次 pre-grad 的输入和输出都是 FX `GraphModule`，但中间节点已经改变。它只改变图中的计算关系；Lowering、Scheduler 和 Codegen 随后才决定这些关系最终形成几个机器代码 kernel。

## pre-grad 为什么更难安全改图

pre-grad 收到的图仍可能直接包含用户写下的原地操作。先看一张没有命中任何优化规则的图：

```python
def f(x):
    old = torch.sin(x)
    x.add_(1)
    return old, x
```

`sin` 要读取写入前的 x，`add_` 随后计算 `x + 1`，并把结果写回调用者传入的那块存储。两者都以 x 为输入，但普通的“谁读取谁的结果”并没有画出这条存储读写顺序：

```text
数据引用：x ──> sin
          x ──> add_

还必须保留的存储顺序：sin 读旧 x ──> add_ 写 x
```

假如一条 pass 移动了相关节点，使 `add_` 先执行，`old` 就会读到修改后的值。这里的例子不是 pre-grad 自动执行的第二种优化，而是对所有相关改写的安全要求。

共享存储会让这项要求跨越不同的 Tensor 名字：

```python
v = x.view(-1)
old = torch.sin(v)
x.add_(1)
later = torch.cos(v)
```

`v` 与 x 指向同一块存储。`old` 应看到写入前的值，`later` 应看到写入后的值。源码把修改已有数据称为 mutation，把两个 Tensor 指向同一块存储称为 alias。改动这些节点或重排它们的 pass，必须同时保留值结果、存储共享关系和读写次序。与这些值无关的 pass 不需要凭空处理所有 mutation；要求针对的是改写可能影响的存储关系。

这就是源码建议优先把新规则写在后两个入口的原因：pre-grad 的图尚未 functionalize，也尚未统一各种参数写法，作者必须自行证明规则对 alias、mutation 和不同调用形式都安全。例如调用者既可能写 `relu(x, False)`，也可能写 `relu(x, inplace=False)`；pre-grad matcher 若关心第二个参数，就要正确处理这些写法。batch ReLU 主案例选择 `inplace=False`，正好避开了输入写回。

## AOTAutograd 把图改成后两个入口使用的形式

AOTAutograd 位于 pre-grad 之后。对刚才的原地修改案例，它把“计算新值”和“让调用者观察到输入已更新”分开：

```text
pre-grad 出口：
v = view(x, [-1])
old = sin(v)
add_(x, 1)
later = cos(v)

AOTAutograd 处理后：
v = aten.view(x, [-1])
old = aten.sin(v)
new = aten.add(x, 1)       # 普通计算，不在这里覆盖 x
aten.copy_(x, new)         # 在原来的语义位置写回调用者的 x
later = aten.cos(v)
```

这个过程称为 functionalization。它使内部的新值计算采用函数式操作；如果 Python 函数本来就会修改调用者的输入，仍需用显式 `copy_` 保留调用者能够观察到的写回。因此本文后面说“函数化后的图”，意思是新值计算已经拆成普通操作，不是说整张图绝不会出现任何写操作。`copy_` 仍在 `sin(v)` 和 `cos(v)` 之间，v 仍与 x 共享存储，所以两次读取继续分别看到旧值和新值。

AOTAutograd 还把操作整理成 `aten.*` 调用。训练时，autograd 根据前向计算加入求梯度所需的节点，暂时形成一张同时含前向与反向工作的 joint graph；partition 随后把它拆成前向图和反向图。推理没有反向图，也不执行这次拆分。

三个入口因此位于下面的位置：

```text
Dynamo FX GraphModule
        │
        ▼
pre_grad_passes
        │
        ▼
AOTAutograd：functionalization，并整理成 ATen 调用
        │
        ├─ 训练：形成 joint graph ─> joint_graph_passes ─> partition
        │                                                ├─ 前向图 ─┐
        │                                                └─ 反向图 ─┤
        │                                                            │
        └─ 推理：一张函数化的图 ──> joint_graph_passes ──────────────┤
                                                                     ▼
                                                          每张图运行 post_grad_passes
                                                                     ▼
                                                               GraphLowering
```

`joint_graph_passes` 是函数名。训练时它收到真正的 joint graph；推理复用这个函数，但收到的是一张函数化的推理图。函数名来自训练路径，不表示推理也凭空产生了反向节点。

## joint 入口在拆图前处理什么

训练时，joint 入口是 partition 前最后一次修改整张 joint graph 的位置。一条规则如果确实要同时查看前向结果和反向使用者，只能在拆分前取得这两部分关系。当前实现也在这里做常量折叠和局部恒等式清理。

例如 `y = exp(x)` 的反向计算会使用前向结果 y：给定上游梯度 `grad_y`，其中一项关系是 `grad_x = grad_y * y`。在 joint graph 中，产生 y 的前向 `exp` 与读取 y 的反向乘法同时存在：

```text
前向：x ─> exp ─> y ─────────────> 前向输出
                  └─> mul(grad_y, y) ─> grad_x
                      反向计算
```

如果用户通过当前源码提供的 joint custom pass 编写一条成对改写规则，要求同时匹配前向 `exp` 和这个反向使用者，那么 pre-grad 时反向节点尚未生成，post-grad 时两部分已经被 partition 分开；joint 是三处中唯一拥有这对关系的入口。这个例子说明怎样按所需信息选择入口。当前内置的 `add(x, zero)` 清理使用的范围更小。

当前默认清理中的一个具体改写并不需要跨越前向与反向。常量折叠先确定一项 Tensor 计算结果全为 0；当 `zero` 与 x 的 shape、dtype、device 相同且 `alpha=1` 时，后续清理把：

```text
x ─> aten.add(x, zero) ─> consumer
```

改成：

```text
x ─────────────────────> consumer
```

pass 让 `consumer` 直接读取 x，再删除无人使用的 add。这说明当前 joint 入口确实会改节点，但它不是“必须同时看到前向和反向”的示例。源码提供了 partition 前的完整图范围；某条具体规则是否使用这项能力，要看该规则本身。推理复用同一个入口，是为了在其单图上运行这组同样适用的函数化图 pass，而不是为了查看不存在的反向图。

训练图完成 joint passes 后才由 partition 拆成前向图和反向图。推理图不拆分，直接继续。

## post-grad 入口逐张处理最终交给 Lowering 的图

`post_grad_passes` 每次收到一张已经函数化、使用 ATen 调用的图。训练时前向图和反向图分别调用一次；推理时只有一张图。这是进入 `GraphLowering` 前的最后一个通用 FX pass 入口。

当前实现会在这里运行面向单图的清理和 pattern replacement。完成这些修改后，它整理节点顺序，并在末尾尝试把不会破坏其他读取结果的函数式操作改回原地形式；源码把这一步称为 reinplace。reinplace 故意放在最后，因为它会重新引入 mutation，前面的清理和匹配可以先在函数式图上工作。

例如当前空操作清理可以删除不改变 dtype 的转换。假设 x 是图内产生的 float32 中间结果，转换结果只供 `mul` 使用，图也不把转换节点单独作为输出：

```text
改写前：x(float32) ─> convert_element_type(x, float32) ─> mul
改写后：x(float32) ────────────────────────────────────> mul
```

在源码检查的 shape、dtype、device、stride 等条件相同后，pass 让 `mul` 直接读取 x，并删除无用的转换节点。这里的例子展示 post-grad 实际改图，不表示这种局部等价变换从理论上只能写在 post-grad。它适合这个入口，是因为规则面对的正是已函数化、已统一成 ATen 调用、即将逐张交给 Lowering 的图。

post-grad 还能直接检查这一张图的输入和输出边界。以一个值相同但原本分配新存储的 `clone` 为例：

```text
情况一：x(函数输入) ─> clone ─> output
情况二：x(函数输入) ─> clone ─> mul ─> output
```

在情况一中，若直接删除 clone，output 会从一份新存储变成与输入 x 共享存储，函数对外的 alias 关系发生变化，所以当前空操作清理会保留它。在情况二中，clone 只是图内中间值；满足其余元信息条件时，让 mul 直接读取 x 不会让函数输出与输入新增共享关系，清理才可以删除 clone。post-grad 已经分别拿到训练前向图、训练反向图或推理图，因此能针对当前这张图的 inputs 和 outputs 作出判断。

## 怎样判断一条规则能在哪个入口运行

三个入口不是按“融合”“删除”这类优化名称分工。判断时要依次看规则实际需要的图：

| 问题 | 对入口选择的影响 |
| --- | --- |
| matcher 必须识别 `torch.nn.functional.*`、原始 module 或用户调用形式吗？ | 需要这些尚未整理的写法时，选择 pre-grad，并自行保证 alias、mutation 和参数形式安全。 |
| 规则可以等到函数化和 ATen 调用形式出现吗？ | 可以在 joint 或 post-grad 考虑；通常不必承担 pre-grad 的额外安全负担。 |
| 训练规则必须同时查看 partition 前的前向与反向关系吗？ | 需要时选择 joint；拆分后任一单图都不再拥有完整关系。 |
| 规则要分别处理最终前向图、反向图或推理图，或依赖它们各自的边界吗？ | 选择 post-grad；它每次处理一张将交给 Lowering 的图。 |
| joint 和 post-grad 都拥有规则需要的信息吗？ | 本篇只能把候选入口缩小到这两处，不能仅凭优化名称推出唯一答案。最终还要到当前源码查看注册位置、训练/推理覆盖范围、与相邻 pass 的顺序及是否会重复匹配。 |

batch ReLU 展示了最后一种情况的一部分。pre-grad 版本匹配 `torch.nn.functional.relu`，post-grad 版本匹配 `aten.relu.default`；当前默认配置启用前者，所以本篇案例在 pre-grad 改图。换句话说，节点写法决定某个 matcher 能不能工作，当前注册和配置再决定这次编译实际在哪个入口改图。不能只看到“这是融合”就推出它必然属于某个入口。

把本篇的三个实际改写放回流程中：

| 入口 | 收到的图 | 本篇展示的实际改动 | 下一站 |
| --- | --- | --- | --- |
| pre-grad | Dynamo 的 FX 图，可能仍有高层调用和原地操作 | 五个 `F.relu` 改成 stack、一次 ReLU、unbind/getitem | AOTAutograd |
| joint | 函数化后；训练为 partition 前 joint graph，推理为单图 | consumer 绕过 `add(x, zero)`，删除 add | 训练进入 partition；推理继续 post-grad |
| post-grad | partition 后的前向/反向单图，或推理单图 | mul 绕过同 dtype 空转换，删除转换节点 | GraphLowering |

某个入口没有规则命中时，计算关系在该入口保持不变。下一篇[递归进入控制流子图](01-recursive-entry/README.md)继续说明图中含 `cond` 分支或 `while_loop` 循环体时，这三个入口怎样进入其中保存的子图。
