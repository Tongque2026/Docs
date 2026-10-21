# 图优化：在进入 Inductor IR 前整理 FX 图

[返回总览](../README.md) · 下一篇：[Lowering](../02-lowering/README.md)

图优化检查同一项计算能否用更规整的节点关系表示。它会修改 FX 节点和边；处理完成后，交给 Lowering 的仍是一张可执行的 FX `GraphModule`，还不是 kernel。

## 为什么要在三个位置分别改图

用一份会让差异显现的请求：

```python
def fn(x):
    old = torch.sin(x)
    x.add_(1)
    return old, x
```

输入 `x=[0,1,2]`。请求要求 `old` 读取修改前的 x，同时调用者持有的 x 最终变成 `[1,2,3]`。独立实验 [`experiments/16_pass_stage_graphs.py`](../experiments/16_pass_stage_graphs.py) 在三个 hook 取得以下图：

| 编译阶段 | 图里的关键关系 | 为什么此时能这样表达 |
| --- | --- | --- |
| pre-grad | `old=sin(x)`，随后 `x.add_(1)` | 图中仍直接保留原地算子，以及多个引用可能共用同一块存储的关系 |
| joint | `old=sin(x)`，`new=add(x,1)`，`copy_(x,new)` | AOTAutograd 已把数值计算写成函数式 add，再用显式 copy_ 表达调用者可见的写回 |
| post-grad | 本例仍为 `sin → add → copy_` | 送入单图 Lowering 前，计算值与写回副作用已经分开表示 |

对具体输入，`sin` 必须先读取 `[0,1,2]`，得到 `[0,sin(1),sin(2)]`；函数式 add 得到 `[1,2,3]`，copy_ 再把它写回原 x。实验同时观察到返回的 old 是修改前的正弦，调用者的 x 是 `[1,2,3]`。

源码把“一次遍历图，并按一组规则检查或修改节点”的工作称为一项 pass。pass 不能在三个位置之间随意搬动，因为同一个操作走到不同位置时，图里的写法已经变了：最开始还是 `add_`；经过 AOTAutograd 后，变成了普通 `add` 加一次 `copy_`。只会处理普通 `add` 的规则必须等到拆分完成后再运行；要检查用户是否写了 `add_` 的规则，则必须在拆分前运行。

## 把三个阶段放回编译流程

现在再给三个位置命名。这里的“阶段”不是三个新组件，而是同一张图在编译流程中先后经过的位置：

```text
Dynamo 交出的图，其中还可能直接出现 x.add_(1)
  └─ pre_grad_passes
       └─ AOTAutograd 把 add_ 拆成 add 和 copy_
            └─ 训练时把前向与反向放在一张 joint graph 中检查
                 └─ joint_graph_passes
                      └─ partition 拆出分别编译的前向图和反向图
                           └─ 每张单图运行 post_grad_passes
                                └─ GraphLowering
```

`pre_grad_passes` 还能直接看见原地修改，以及多个张量共用一块存储的关系。训练时，`joint_graph_passes` 能同时看见某个前向结果和反向计算对它的使用。`post_grad_passes` 每次只处理一张已经拆开的前向图或反向图，其中原地数值计算与写回已经分开表示。

所以，图优化不是只运行一次。编译器会依次在三个位置改图：

- `pre-grad` 仍能看见用户写下的 `add_`，因此这里可以处理原地操作和共享存储；
- `joint` 同时含有训练的前向计算和反向计算，因此这里可以决定哪些前向结果留给反向、哪些结果以后重算；
- `post-grad` 每次只拿到一张拆好的前向图或反向图，而且原来的 `add_` 已写成 `add + copy_`，因此这里可以按普通 ATen 节点做删除、替换和重新排序。

这三个名称只标记改图发生在编译流程的哪个位置。它们不是三种抽象的“优化语义”。

## 本层三项直接职责

改图后还有三份信息必须同步。新节点要排在使用它的节点之前；每个节点旁的 shape、dtype、device、stride 记录要对应新结果；`GraphModule.forward` 的 Python 代码也要重新生成。源码分别用稳定拓扑排序、更新 `node.meta`、`recompile()` 和 `lint()` 完成这些检查。这里的 `lint()` 会在引用了尚未定义的节点、节点属于错误的图等情况下报错。

```text
输入 FX 图
   │
   ├─ 选择改图位置：处理 add_ 的规则放在 pre-grad；需要同时查看前向和反向的规则放在 joint；只处理拆分后单图的规则放在 post-grad
   ├─ 完成模式替换：匹配子图并换成更合适的整体表示，同时把原先读取旧结果的节点改读新结果
   └─ 让改后的图重新可执行：排列节点、更新 shape/dtype 等记录、重建 forward 并检查引用
   │
   ▼
仍具有 FX 节点、边和节点 meta 的 GraphModule
```

第一项决定一项 pass 应在原地写回（源码称 mutation）和共享存储（源码称 alias）仍存在时运行，还是等 `add_` 已改成 `add+copy_` 之后再运行；规范化、删除无用节点、把安全的函数式操作改回原地形式（reinplace）等工作都由相应阶段组织。第二项改变 Lowering 将看到的节点和边。第三项保证这种修改没有留下错误节点顺序、过期 shape/dtype 记录或未更新的 `forward`。三项分别由后续三个子篇打开，目录树与这里的职责保持一致。

## 案例一：逐元素链没有在 FX 层融合

贯穿案例的 post-grad 图仍是：

```text
add → relu → mul → output
```

随后生成代码却只有一个 `cpp_fused_add_mul_relu_0`。因此本例的阶段判断是：

| 观察位置 | 实际对象 | 观察结果 |
| --- | --- | --- |
| post-grad pass 出口 | FX nodes | add、relu、mul 三个节点 |
| Scheduler 之后 | fused scheduler work | 一个融合工作项 |
| CPU codegen 出口 | C++ function | 一个循环完成三项运算 |

如果只看最终 kernel 名称里有 `fused`，不能反推出 FX 图已经被替换；要看对应阶段的对象。

## 案例二：一次真实规则怎样改边和节点

先拆开一条 FX 节点记录。`a = x + y` 的节点包含 `op="call_function"`（调用函数）、`target=aten.add.Tensor`（具体调用谁）和 `args=(x,y)`（输入引用）。后文替换节点时会改 target 或 args，但必须保持这些引用指向合法节点。

编译期还要判断 x 的 dtype，却不需要为 x 准备真实元素。PyTorch 用 FakeTensor 保存 x 的 shape、dtype、device 和 stride，而不分配对应数据；节点执行后的 FakeTensor 通常放在 `node.meta["val"]`。例如 float32 的 x、y 经过 add 后，新节点的 meta 里仍得到 shape `[3]`、dtype float32 的 FakeTensor。模式条件可读取这份 dtype，替换建立的新节点也要重新产生对应 meta。

独立实验为当前 `PatternMatcherPass` 注册了下面的规则，用它专门观察图替换机制：

```text
匹配：mul(add(x,y), 2)
替换：add(mul(x,2), mul(y,2))
条件：x 的 fake tensor dtype 必须是 float32
```

输入 `x=[1,2,3]`、`y=[10,20,30]` 时，两边都得到 `[22,44,66]`。float32 图的实际变化是：

```text
改写前：x,y → add(a) → mul(a,2)(b) → output(b)

捕获：x、y 绑定到规则的两个参数。
新建：mul(x,2)(m0)、mul(y,2)(m1)、add(m0,m1)(d)。
重接：output 原来读取 b，现在读取 d。
删除：b 已无使用者；a 也已无其他使用者，因此一并删除。

改写后：x → mul(m0) ─┐
                       ├→ add(d) → output(d)
          y → mul(m1) ─┘
```

实验记录 `float32 matches=1`，并在 `lint/recompile` 后验证新图与原函数数值相同。把输入改成 int64 时，结构仍相同，但 `extra_check` 拒绝：`matches=0`，原来的 add→mul 图保留。这给“条件满足/不满足”一份具体内容，而不是把关键分支藏在概括里。

本层完整交接是：

```text
原 FX 子图及 node.meta
  → matcher 检查结构和条件
  → replacement 建立新节点并重接 users
  → 更新 fake tensor/meta，稳定拓扑，recompile/lint
  → GraphLowering 接收新图
```

下一层会继续解释 matcher 怎样先按末端节点缩小候选范围、怎样从输出沿 args 逐层核对 add 和 mul，以及怎样把替换图中的新节点复制进原图。本层先掌握：图替换真实改变了 Lowering 将看到的节点和边，而 scheduler fusion 不改 FX 图。

如果旧 add 还有另一条边直接通向第二个输出，replacement 只接管旧 mul 的结果：output0 改读新 add，output1 仍读旧 add。旧 mul 无 user 后可以删除，旧 add 仍有 output1 这个 user，不能为了清除匹配子图而强删。替换的边界由实际 users 决定，不由“匹配曾覆盖哪些节点”单独决定。

## 图改完以后，四份信息必须重新一致

FX 图不只是屏幕上画出的节点集合。一次编辑至少要保持下面四件事一致：

| 要维护的内容 | 若过期会怎样 | 本层交给 Lowering 的结果 |
| --- | --- | --- |
| 拓扑顺序 | consumer 可能排在 producer 前，解释图时先用后定义 | 每个节点都排在其依赖之后 |
| node meta / fake tensor 属性 | 后续按旧 shape、dtype 或 stride 作决定 | 改写后的节点拥有与新计算相符的属性 |
| GraphModule 的 `forward` | 图对象已改，Python 调用仍执行旧代码 | `recompile()` 让可执行代码反映新图 |
| 图合法性 | 悬空引用、错误 owner 或顺序问题可能继续潜伏 | `lint()` 在进入 Lowering 前检查节点引用和所属图是否合法 |

独立实验 [`experiments/03_graph_invariants.py`](../experiments/03_graph_invariants.py) 把 x 故意移动到使用它的 add 后面：

```text
错误顺序：y, add(y,x), x
lint：add 使用了尚未定义的 x
stable_topological_sort 后：y, x, add(y,x)
```

排序只修复依赖顺序，不改变 add 仍读取 y、x。实验的第二部分把一张 `a+b` 图中节点 target 改成乘法。修改 Graph 后、调用 `recompile()` 前，`gm(2,3)` 仍执行旧 `forward`，结果是 5；recompile 后生成代码变为 `a*b`，结果才是 6。由此可见，“节点已经改了”与“GraphModule 已能执行新图”是两个连续步骤。

meta 更新处理的是另一个维度：若 replacement 改变 shape、dtype 或 stride，后续 pass 和 Lowering 必须读取新结果的属性。Q13 会打开当前实现怎样增量传播 fake tensor 属性；本层只需记住其消费者——这些属性是下一阶段选择计算和布局表示的输入。

## 失败或不改写也是正常出口

一个候选模式可能因为 dtype、device、shape、stride、训练/推理状态或配置不满足而不应用。此时图并没有“优化失败”：原 FX 节点继续交给 Lowering。若该算子有直接 lowering，便直接建立 Inductor IR；若有 decomposition，先把一个大算子拆成若干已有小算子再分别 Lower；若只能 fallback，就保留算子身份，运行时由 wrapper 调用现成的 PyTorch 实现。

同样，删除无用节点、重新排列节点、重新引入原地写入都不能只看节点名字，还要保证读写顺序、共享存储和返回结果不变。最后一项在源码中称为 reinplace：例如前面已被拆成 `new=add(x,1); copy_(x,new)` 的写回，在确认不会破坏其他旧值读者后，可以改成直接对目标存储执行原地操作。`post_grad_passes` 最后执行稳定拓扑、更新 fake tensor 元信息、重新编译 GraphModule 并 `lint()`，让改过的图成为 Lowering 可可靠读取的输入。

它们不能互相替代：拓扑正确不代表 `forward` 已更新，recompile 成功也不代表 node meta 已与新计算一致。

## 把三项职责接成一次完整判断

假设 post-grad 图中出现 float32 的 `mul(add(x,y),2)`，并且 add 还作为第二项输出：

1. 当前所处阶段先说明图已经 functionalized，规则可以按纯函数 add/mul 理解，不必把这里的 add 当成 `add_` 写入。
2. pattern replacement 检查结构和 float32 条件，建立两项 mul 与新 add，只把原 mul 的外部 user 重接到新 add；旧 add 因第二项输出仍有 user 而保留。
3. 不变量维护把新旧节点排成 producer 先于 consumer，更新新增节点的 meta，重新生成 GraphModule `forward`，再检查图合法性。
4. GraphLowering 最终收到的是“新分配律结果 + 仍需返回的旧 add”这张合法 FX 图，而不是 matcher 中途的临时节点集合。

若把输入改成 int64，第 2 步的 extra_check 拒绝，后续不变量维护仍可以检查原图并把它交给 Lowering。拒绝一项优化不会改变原程序要计算的结果，也不是编译失败。

## 读完后应能判断

如果你看到 FX 图中还有三项逐元素运算，但只有一个 kernel，应先归因于 Scheduler/后端融合；如果 FX 图中已经出现一个专用算子，则图优化确实改变了 Lowering 的输入。下一篇开始打开 Lowering 如何解释这张图。

继续深入：

1. [pre-grad、joint、post-grad 分别处理什么](01-pass-stages/README.md)
2. [Pattern Matcher 怎样完成一次安全替换](02-pattern-matching/README.md)
3. [图改写后怎样恢复 FX 不变量](03-graph-invariants/README.md)
