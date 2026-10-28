# 三个 pass 入口分别接住哪一张图

[返回图优化](../README.md) · [返回总览](../../README.md)

`torch.compile` 中，Dynamo 先把一次 Python 执行记录成 FX 图。图中的一项操作是一个节点；一个节点读取另一节点的结果，就形成一条引用。`GraphModule` 保存这张图，并提供按照节点关系执行的 `forward`。

同一个 Python 函数在编译过程中会换成不同的图表示。以 `old = sin(x); x.add_(1)` 为例，图会依次出现下面四种状态：

```text
Dynamo 刚记录完：       sin(x)；add_(x, 1)
AOTAutograd 整理后：    sin(x)；new = aten.add(x, 1)；aten.copy_(x, new)
训练图加入求梯度后：    前向节点和反向节点暂时在同一张图里
训练图拆分后：          独立的前向图、独立的反向图
```

第一行仍保留用户写下的高层调用和原地操作；第二行把“算出新值”和“写回 x”分成了两个节点；第三行能同时看见某个前向结果及其反向使用者；第四行则只剩一张即将单独编译的前向图或反向图。推理没有反向计算，因此不会产生最后这次拆分，整理后始终是一张图。

Inductor 在这些变化之间安排三次改图机会：

```text
Dynamo 图 ─> pre-grad ─> AOTAutograd 整理并在训练时加入反向
                              ├─ 训练：joint graph ─> joint ─> 拆成前向图和反向图 ─┐
                              └─ 推理：函数化单图 ─> joint ──────────────────────┤
                                                                                 ▼
                                                         每张图运行 post-grad ─> Lowering
```

pre-grad 能处理尚未整理的原始调用；joint 能在训练图拆开前同时查看前向和反向关系；post-grad 逐张处理最终要交给 Lowering 的图。每个入口都会按顺序调用一组 pass。这里的 pass 是“一轮检查图、在条件满足时修改节点和引用的程序”：它可以是普通 Python 函数，也可以是 `PatternMatcherPass` 对象的 `apply` 方法。

`PatternMatcherPass` 对象保存一组已经注册的局部改写规则。阶段入口拿到当前图后，调用其中一个对象的 `apply(graph)`；该对象在这张图中尝试自己保存的规则，命中时直接修改图，最后返回成功应用的 entry 数量。一个阶段可以调用多个这样的对象，也可以在它们前后运行其他普通 pass。因此，pre-grad、joint、post-grad 是三处阶段入口，`PatternMatcherPass` 是这些入口可以调用的一种具体改图对象。

下面先走完 pre-grad 的一次实际改写。

## pre-grad 怎样完成一次改图

`pre_grad_passes` 接收 Dynamo 产生的 FX `GraphModule`，按当前配置依次调用 pre-grad pass。一条 pass 命中时，它会新增、删除或重接节点；一条 pass 没找到候选时，这条 pass 不改图，入口继续运行后面的工作。所有配置步骤都没有改变计算关系时，原来的计算关系保持不变。入口末尾会整理节点顺序、检查节点引用，并依据当前图重新生成 `GraphModule.forward`。整理后的 `GraphModule` 随后交给 AOTAutograd。

下面用当前默认启用的 `batch_relu` 走完一次实际改写。假设图里有五个互不依赖的 ReLU：

```text
a0(shape=[2, 3]) ──> relu0 ──> use0
a1(shape=[2, 3]) ──> relu1 ──> use1
a2(shape=[2, 3]) ──> relu2 ──> use2
a3(shape=[2, 3]) ──> relu3 ──> use3
a4(shape=[2, 3]) ──> relu4 ──> use4
```

五个节点调用的都是 `torch.nn.functional.relu(..., inplace=False)`。ReLU 逐元素把负数变成 0，其他数保持不变。把五个 `[2, 3]` Tensor 叠成 `[5, 2, 3]` 后做一次 ReLU，再沿第 0 维拆开，五份结果分别与原来的 `relu0` 至 `relu4` 相同。

规则按输入 shape 和 `inplace` 参数分组。shape 相同才能叠成规则预期的批量输入，`inplace=False` 保证旧节点不会修改各自的输入。五条路径互不读取彼此的结果，所以合并不会引入原图没有的先后依赖。shape、`inplace` 和路径间依赖决定替换能否保持原计算关系；当前配置要求至少有五个候选节点，这个数量门槛只决定本次是否执行合并。少于五个候选时，这条规则保留原图。

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

寻找候选节点、检查 shape、`inplace`、路径关系和数量条件的部分称为 matcher；命中后执行上面四步的代码负责 replacement。这个 batch ReLU pass 展示的是 matcher 与 replacement 的职责，不要求它一定由 `PatternMatcherPass` 类实现。pre-grad 入口依次运行这些具体 pass，最后完成图的整理和检查。

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

假如一条 pass 移动了相关节点，使 `add_` 先执行，`old` 就会读到修改后的值。这个例子说明相关改写必须保留的安全要求：`sin` 读取旧值之后，`add_` 才能写入 x。

共享存储会让这项要求跨越不同的 Tensor 名字：

```python
v = x.view(-1)
old = torch.sin(v)
x.add_(1)
later = torch.cos(v)
```

`v` 与 x 指向同一块存储。`old` 应看到写入前的值，`later` 应看到写入后的值。源码把修改已有数据称为 mutation，把两个 Tensor 指向同一块存储称为 alias。改动这些节点或重排它们的 pass，必须同时保留值结果、存储共享关系和读写次序；规则只需检查自身改写会触及的这些关系。

pre-grad 的图尚未把原地操作拆成“计算新值”和“显式写回”，也尚未把各种参数写法统一起来，因此这里的规则要自行保证 alias、mutation 和不同调用形式都安全。例如调用者既可能写 `relu(x, False)`，也可能写 `relu(x, inplace=False)`；matcher 若关心第二个参数，就要同时识别这些写法。batch ReLU 主案例只接收 `inplace=False` 的节点，因而不会把五次输入写回错误地合成一次批量写回。

本例在 pre-grad 命中，是因为当前配置启用了识别 `torch.nn.functional.relu` 的规则，而且五个节点都满足它的安全条件。AOTAutograd 整理节点以后，另一个 post-grad 实现可以匹配 `aten.relu.default`。当前启用的规则和当时图中的节点写法共同决定本次改写发生在哪一处。

## AOTAutograd 把图改成后两个入口使用的形式

AOTAutograd 位于 pre-grad 之后。对刚才的原地修改案例，它把“计算新值”和“让调用者观察到输入已更新”分开：

```text
pre-grad 出口：
v = view(x, [-1])
old = sin(v)
add_(x, 1)
later = cos(v)

AOTAutograd 处理后：
old_view = aten.view(x, [-1])
old = aten.sin(old_view)
new = aten.add(x, 1)             # 普通计算，不覆盖 x
new_view = aten.view(new, [-1])
later = aten.cos(new_view)        # 读取函数式的新值
aten.copy_(x, new)                # 函数出口前兑现调用者可见的输入修改
```

这个过程称为 functionalization。图内不再让 `later` 依靠一次中途写回读取新值：
`old_view` 继续引用旧 x，`new_view` 改为引用函数式计算得到的 `new`，所以 old 和 later
分别读取修改前后的两个值版本。原 Python 函数还要求调用者最终看到 x 已更新，因此图在
返回前用 `copy_(x,new)` 把新值写回输入 storage。图内值版本传播和函数出口兑现输入修改
是两件工作。

AOTAutograd 还把操作整理成 `aten.*` 调用。训练时，autograd 根据前向计算加入求梯度所需的节点，暂时形成一张同时含前向与反向工作的 joint graph；partition 随后把它拆成前向图和反向图。推理没有反向图，也不执行这次拆分。

把上述变化补全到编译流程中：

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

`joint_graph_passes` 是训练和推理共同调用的函数名。训练时它收到含前向和反向计算的 joint graph；推理时它收到一张函数化的推理图。

## joint 入口能看见前向和反向之间的关系

训练时，joint 入口是拆图前最后一次修改整张 joint graph 的位置。一条规则如果要同时查看前向结果和反向使用者，只能在这里取得两部分关系。先看这种关系具体是什么。

例如 `y = exp(x)` 的反向计算会使用前向结果 y：给定上游梯度 `grad_y`，其中一项关系是 `grad_x = grad_y * y`。在 joint graph 中，产生 y 的前向 `exp` 与读取 y 的反向乘法同时存在：

```text
前向：x ─> exp ─> y ─────────────> 前向输出
                  └─> mul(grad_y, y) ─> grad_x
                      反向计算
```

一条需要同时检查前向 `exp` 和反向 `mul(grad_y, y)` 的自定义规则，必须在 joint 入口运行：pre-grad 时反向节点尚未生成，拆图后的 post-grad 又只能看见其中一张图。当前源码提供的 joint custom pass 接口可以接收这类规则。

当前默认 joint pass 的实际改图范围更小。它不需要前向和反向同时存在，而是在 joint 入口中清理局部恒等关系。下面走完其中一次节点删除。

当前默认清理还会处理局部关系。常量折叠先确定一项 Tensor 计算结果全为 0；当 `zero` 与 x 的 shape、dtype、device 相同且 `alpha=1` 时，后续清理把：

```text
x ─> aten.add(x, zero) ─> consumer
```

改成：

```text
x ─────────────────────> consumer
```

pass 让 `consumer` 直接读取 x，再删除无人使用的 add。训练时，改后的整张图随后才被拆成前向图和反向图；推理时，改后的单图直接继续到 post-grad。

joint graph 提供了 `exp` 与其反向使用者之间的完整关系。当前默认 pass 则利用其中的局部关系删除 `add(x, zero)`；它在 joint 运行是当前注册和 pass 顺序的结果。这两项事实分别说明这个入口可提供的信息范围，以及当前编译实际在这里完成的改图。

训练图完成 joint passes 后才由 partition 拆成前向图和反向图。推理图不拆分，直接继续。

## post-grad 入口逐张处理最终交给 Lowering 的图

`post_grad_passes` 每次收到一张已经函数化、使用 ATen 调用的图。训练时前向图和反向图分别调用一次；推理时只有一张图。这是进入 `GraphLowering` 前的最后一个通用 FX pass 入口。

当前源码在 post-grad 模块中长期保存了多个 `PatternMatcherPass` 对象。注册代码把规则放入指定对象；运行到 post-grad 时，入口按顺序遍历这些对象并调用各自的 `apply(gm.graph)`。所以“一条规则在 post-grad 运行”的完整含义是：该规则注册在 post-grad 会调用的某个对象里，而 post-grad 把当前这张单图交给同一个对象执行。

当前实现会在这里运行面向单图的清理和 pattern replacement。完成这些修改后，它整理节点顺序，并在末尾尝试把不会破坏其他读取结果的函数式操作改回原地形式；源码把这一步称为 reinplace。它放在末尾，因为这一步重新引入 mutation，前面的清理和匹配可以先处理没有原地写入的节点关系。本篇只解释它在流程中的位置；哪些读取关系允许 reinplace，由相应实现篇展开。

例如当前空操作清理可以删除不改变 dtype 的转换。假设 x 是图内产生的 float32 中间结果，转换结果只供 `mul` 使用，图也不把转换节点单独作为输出：

```text
改写前：x(float32) ─> convert_element_type(x, float32) ─> mul
改写后：x(float32) ────────────────────────────────────> mul
```

在源码检查的 shape、dtype、device、stride 等条件相同后，pass 让 `mul` 直接读取 x，并删除无用的转换节点。当前规则在 post-grad 面对已函数化、已统一成 ATen 调用、即将逐张交给 Lowering 的图。

post-grad 还能直接检查这一张图的输入和输出边界。以一个值相同但原本分配新存储的 `clone` 为例：

```text
情况一：x(函数输入) ─> clone ─> output
情况二：x(函数输入) ─> clone ─> mul ─> output
```

在情况一中，若直接删除 clone，output 会从一份新存储变成与输入 x 共享存储，函数对外的 alias 关系发生变化，所以当前空操作清理会保留它。在情况二中，clone 只是图内中间值；满足其余元信息条件时，让 mul 直接读取 x 不会让函数输出与输入新增共享关系，清理才可以删除 clone。post-grad 已经分别拿到训练前向图、训练反向图或推理图，因此能针对当前这张图的 inputs 和 outputs 作出判断。

## 怎样判断一条规则能在哪个入口运行

入口由规则需要读取的图表示和节点关系决定。判断时依次回答下面几个问题：

| 问题 | 对入口选择的影响 |
| --- | --- |
| matcher 必须识别 `torch.nn.functional.*`、原始 module 或用户调用形式吗？ | 需要这些尚未整理的写法时，选择 pre-grad，并自行保证 alias、mutation 和参数形式安全。 |
| 规则可以等到函数化和 ATen 调用形式出现吗？ | 可以在 joint 或 post-grad 考虑；通常不必承担 pre-grad 的额外安全负担。 |
| 训练规则必须同时查看 partition 前的前向与反向关系吗？ | 需要时选择 joint；拆分后任一单图都不再拥有完整关系。 |
| 规则要分别处理最终前向图、反向图或推理图，或依赖它们各自的边界吗？ | 选择 post-grad；它每次处理一张将交给 Lowering 的图。 |
| joint 和 post-grad 都拥有规则需要的信息吗？ | 两处都是候选；再查看当前源码中的注册位置、训练/推理覆盖范围、相邻 pass 顺序和重复匹配情况。 |

batch ReLU 同时有匹配 `torch.nn.functional.relu` 的 pre-grad 版本和匹配 `aten.relu.default` 的 post-grad 版本；当前默认配置启用前者，所以本篇案例在 pre-grad 改图。节点写法决定 matcher 能否命中，当前注册和配置决定这次编译实际在哪个入口运行它。

把本篇的三个实际改写放回流程中：

| 入口 | 收到的图 | 本篇展示的实际改动 | 下一站 |
| --- | --- | --- | --- |
| pre-grad | Dynamo 的 FX 图，可能仍有高层调用和原地操作 | 五个 `F.relu` 改成 stack、一次 ReLU、unbind/getitem | AOTAutograd |
| joint | 函数化后；训练为 partition 前 joint graph，推理为单图 | consumer 绕过 `add(x, zero)`，删除 add | 训练进入 partition；推理继续 post-grad |
| post-grad | partition 后的前向/反向单图，或推理单图 | mul 绕过同 dtype 空转换，删除转换节点 | GraphLowering |

“某条匹配规则没有命中”只表示这条规则没有改图，入口仍会继续其余步骤。pre-grad 最后总会整理和检查图并重新生成 `forward`；joint 继续运行其他已配置的清理或规则；post-grad 还会整理节点，并在末尾尝试 reinplace。因此，只有所有步骤都没有改变计算关系时，这个入口的计算关系才保持原样。下一篇[递归进入控制流子图](01-recursive-entry/README.md)继续说明图中含 `cond` 分支或 `while_loop` 循环体时，这三个入口怎样进入其中保存的子图。
