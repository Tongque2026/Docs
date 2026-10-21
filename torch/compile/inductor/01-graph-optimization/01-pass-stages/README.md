# 三个处理阶段：pre-grad、joint、post-grad 分别看见什么

[返回图优化](../README.md) · [返回总览](../../README.md)

实现叶子：[递归调用各子图的完整源码](01-recursive-entry/README.md)

我们沿同一条编译流水线看三个位置：AOTAutograd 之前、前向和反向拆分之前、拆分后的单图进入 Lowering 之前。流程每前进一步，图里保留的信息都会变化。例如，两个张量可能共用同一块存储，修改其中一个也会影响另一个；训练时的前向结果还可能被反向计算使用。改写若漏掉这些关系，哪怕代数结果看起来相同，也可能改变程序。

先用本篇贯穿的原地修改看这条流程。最初的图直接含有 `x.add_(1)`；AOTAutograd 随后把它改写为“计算 `new=x+1`，再用 `copy_(x,new)` 写回 x”。源码把这种“用不修改输入的计算加显式写回来表达原地操作”的变换称为 functionalization。

```text
Dynamo 得到的图
    │
    ├─ pre-grad：AOTAutograd 之前，原地修改与别名仍可能直接存在
    ▼
AOTAutograd functionalization，并在训练时建立 joint graph
    │
    ├─ joint：前向与反向拆分之前，还能同时观察两边
    ▼
partition 得到前向图、反向图（推理只有前向路径）
    │
    ├─ post-grad：每张规范化单图进入 Lowering 之前
    ▼
GraphLowering
```

所以本篇要回答的是：一条改图规则需要看见什么？如果它需要看见 `add_`，就在拆分前运行；如果它需要同时看见前向和反向，就在两者分开前运行；如果它只处理普通 ATen 节点，就等图拆成独立的前向图或反向图后运行。

## 第一阶段：pre-grad 面对尚未 functionalize 的图

`compile_fx` 在进入 AOTAutograd 之前递归调用 `_recursive_pre_grad_passes`。这里还没有完成两类整理：原地 `add_` 尚未拆成 `add+copy_`；同一算子也可能仍以不同参数写法出现，例如参数通过位置传入或用关键字传入。源码分别称这两类整理为 functionalization 和 normalization。因此 pre-grad 规则必须同时处理共享存储、原地写入和不同参数写法。

例如请求包含对输入的原地修改：

```python
def f(x):
    y = x.sin()
    x.add_(1)
    return y, x
```

pre-grad 不能仅依据 `sin` 与 `add` 的代数形式随意换序，因为 `y` 必须读取修改前的 x，而返回的 x 是修改后的版本。这个阶段适合处理需要看原始模块或调用结构的改写。当前源码还会统一部分参数写法，并尝试把结构相同、可以共同执行的一组 batch 操作合并；无论具体规则是哪一种，都必须自己处理原地修改、共享存储和不同参数形式。

独立实验把这同一个请求接到三个 custom hook。pre-grad hook 实际看到：

```python
old = torch.sin(l_x_)
l_x_.add_(1)
return (old,)
```

图中 `add_` 直接修改输入引用 `l_x_`。如果把它移动到 sin 之前，old 会从 `sin([0,1,2])` 变成 `sin([1,2,3])`。因此，即使两个节点之间没有一条表示返回值传递的边，也不能断定它们可以换序：`add_` 对同一块数据的写入同样规定了先后关系。

不论 pattern matcher 是否开启，结尾仍会整理节点顺序。量化模型还会把量化/反量化相关节点移动到它们应当靠近的生产者或使用者位置，源码称为 `quant_lift_up`。随后 `lint()` 检查每个输入是否先定义、节点是否属于当前图等结构条件，失败时立即报错；`recompile()` 根据内存中已经改过的 Graph 重新生成 `GraphModule.forward`。完成后，下一步拿到的是节点顺序、引用关系和 Python `forward` 已同步的 GraphModule。

## 中间桥梁：AOTAutograd 改变了问题形态

AOTAutograd 位于 pre-grad 与后两类 pass 之间。对本教程的阶段理解，它至少带来两项关键变化：

- functionalization 把很多原地修改和共享存储关系拆成普通的数值计算，再用必要的写回操作保留调用者能观察到的修改；
- 训练时建立能表达前向和反向关系的图，再按 partition 形成实际编译的前向、反向图。

因此 post-grad 可以声明输入已 normalized/functionalized，而 pre-grad 不能。这里不是说所有 mutation 都消失：post-grad 末尾还会有意 reinplace 某些操作，输入 mutation 也可能以 `copy_` epilogue 形式重新出现。

同一实验的 joint hook 实际看到：

```python
sin = torch.ops.aten.sin.default(arg0_1)
add = torch.ops.aten.add.Tensor(arg0_1, 1)
torch.ops.aten.copy_.default(arg0_1, add)
return (sin,)
```

原来的 `add_` 被拆成函数式 `add` 和显式写回 `copy_`。sin 与 add 都读取修改前的 `arg0_1`；copy_ 在最后把 add 的结果写回原输入。这张图把“新值怎样计算”和“何时更新外部可见 storage”分开，后续 pass 因而更容易分析普通算子关系。

## 第二阶段：joint pass 能看见前向与反向的共同边界

`joint_graph_passes` 的参数是尚未拆开的前向加反向图。当前顺序可以按作用归成下面几步：`remove_noop_ops` 删除结果等于输入、又没有写入的空操作；uniform-value constant folding 把已知所有元素相同的常量计算直接换成常量；random replacement 把某些随机请求改成适合后续拆分和重算的形式。

```text
custom pre pass
  → remove_noop_ops
  → 可选 uniform-value constant folding
  → 两组 pattern matcher passes
  → 可选 random replacement
  → custom post pass
  → 若有改动：stable_topological_sort + lint + recompile
```

它适合只有在前向和反向仍放在一起时才能识别或安排的变换。随后 `min_cut_rematerialization_partition(..., compiler="inductor")` 决定哪些值保存给反向、哪些可在反向重算。

具体判断题：若前向产生中间值 a，反向需要 a，joint 阶段仍能比较“保存 a”与“以后重算 a”的边界；等前向、反向已经成为两张独立编译图，再只看其中一张就缺少这份共同关系。

推理没有反向 partition，但 `fw_compiler_base` 仍会递归调用 joint passes。这里的 `joint` 是这组 pass 和入口沿用的名字：训练时它能看到尚未拆开的前向与反向关系，推理时也会复用这个入口，但不会凭空产生反向计算。

独立实验把 inference 与 training 分开采集：

| 编译请求 | joint hook 看见什么 | post hook 次数与范围 |
| --- | --- | --- |
| `sin(x)*2`，x 不求梯度 | `sin、mul` 的 inference graph | 1 次：仍是 sin、mul 的前向单图 |
| `sum(sin(x)*x)`，x 求梯度并调用 backward | 一张同时含前向 `sin/mul/sum` 与反向 `expand/mul/cos/add` 的 joint graph | 2 次：forward post 只有 sin/mul/sum；backward post 含 expand、sin、两项 mul、cos、mul、add |

training 路径先在 `partition_fn` 对 joint graph 运行 joint passes，再由 `min_cut_rematerialization_partition(..., compiler="inductor")` 拆分；forward compiler 与 backward compiler 随后分别调用 inner compile，因此各自进入 post-grad。inference 路径不会调用 partition_fn，所以 forward compiler 自己先运行 joint passes，再进入一次 post-grad。两条路线都会经过 joint 这一阶段，但它们使用共同信息的目的和后续图数量不同。

## 第三阶段：post-grad 面对每张规范化单图

`fx_codegen_and_compile` 在创建 `GraphLowering` 前运行 `_recursive_post_grad_passes`。一张前向和一张反向会各自进入这里。先按可观察变化理解：删除无用节点；推理时把可相邻执行的生产者和使用者移近；运行模式与通信改写；把新建常量移动到所需 GPU；刷新 FakeTensor 记录；最后才重新引入安全的原地操作。源码调用顺序是：

```text
可选删除无使用者且无副作用的节点（DCE）/ 推理图局部重排
  → 建立 FakeTensorUpdater
  → custom pre
  → 合并可共同执行的 batch 操作、删除空操作、运行模式替换、可选合并相邻矩阵乘法（b2b gemm）
  → 通信相关 passes
  → custom post
  → stable_topological_sort
  → move_constructors_to_gpu
  → 增量更新 fake tensor meta
  → 把确认安全的普通操作改回原地操作（reinplace）
  → 把 auto-functionalized 包装展开成必要的 clone 和底层写入操作
  → 把符合模式的 FSDP all-gather 改成写入已有输出 buffer 的版本
  → recompile + lint
```

最后一项处理 FSDP 的一种通信图。原图先由 `all_gather_copy_in` 准备通信输入和一块输出
buffer，随后 `all_gather_into_tensor(input, ...)` 另外产生结果。匹配成功后，编译器改用
`all_gather_into_tensor_out(input, ..., out=那块已有的输出buffer)`：通信结果直接写进已准备
的 buffer，不再保留“另产结果”的版本。没有这组节点或运行环境没有相应分布式算子时，
该改写直接返回，其他 post-grad 工作照常继续。

这里的 pattern 可以依赖统一写法的 ATen 节点以及 fake tensor 的 shape、dtype、device 信息。`reinplace` 例如会在确认旧值不再被使用后，把“另算一个结果再复制回来”改成直接写原 buffer。它接近末尾才运行；若提前恢复原地写入，前面的规则就不能再把每个节点当成“只产生新值、不覆盖旧值”的普通计算。

实验的 post-grad custom-post hook 对这个小请求仍看到上面的 `sin → add → copy_` 形式，实际运行得到 old=`sin([0,1,2])`，调用者 x=`[1,2,3]`。这次观察只说明该请求在所取 hook 位置的表示；并不声称所有 joint/post 图逐字相同。训练 partition、后续 reinplace 和不同算子都会改变具体节点。

## 同一个修改请求应放在哪一层

| 修改需要观察什么 | 适合放在哪一阶段 | 原因 |
| --- | --- | --- |
| 原始 module/高层调用结构 | pre-grad | AOTAutograd 后这些结构可能已拆解 |
| 前向和反向之间保存/重算的共同关系 | joint | partition 前信息最完整 |
| 规范化 ATen 子图、确定的 fake shape/dtype | post-grad | functionalization 后更易安全匹配 |
| 循环读写能否合成一个 kernel | Scheduler，不属于这三者 | 需要 Lowering 后的 IR 依赖与后端能力 |

## 子图也会递归处理

三个入口都有 `_recursive_*` 包装。它们先找到 higher-order op 携带的子 GraphModule，对子图执行同类阶段，再处理外层图。这样 `cond`、`while_loop` 等内部计算不会因为嵌套而完全绕开图准备。

三个入口传递的附加信息并不完全相同：

- pre-grad 外图收到 example inputs；递归进入 cond/while 子图时没有对应的独立样例列表，所以明确传 `example_inputs=None`。依赖实际样例执行的 pre pass 必须处理这一条件。
- joint 递归不需要 example inputs；子图和外图都接收尚未拆开的前向/反向关系，改写后必须仍保持两边的输入输出引用合法。
- post-grad 把同一个 `is_inference` 标志传入所有子图。推理时允许根据生产者—使用者关系把可相邻执行的节点移近；训练图不走这项只为推理生成顺序服务的重排。

处理顺序是“先子图、后外图”。例如外图含 `cond(pred,true_graph,false_graph,[x])`，递归入口先分别完成 true/false GraphModule 的当前阶段，再处理仍含 cond 节点的父图。父图 pass 可以依赖子图已经收尾，但递归不会建立一张把两支节点直接摊平的线性图。

递归不等于把子图摊平。外层仍保留控制流节点和子图接口，Lowering 时由对应高阶算子路径接住。

## 没有候选、配置关闭和阶段失败怎样退出

“运行到一个阶段”不保证图一定发生 pattern replacement。三个阶段的控制条件不同：

| 条件 | 当前阶段仍做什么 | 出口 |
| --- | --- | --- |
| pre 的 `pattern_matcher=False` | 跳过 matcher 主体，但 custom pass 若配置仍可运行；随后稳定拓扑、quant lift、lint、recompile | 合法 pre GraphModule |
| joint 没有计入任何修改 | remove-noop 等阶段工作仍经过；只有 `count>0` 时执行末尾 stable sort/lint/recompile 分支 | 原关系或已被阶段工作改变的 joint graph |
| post 的 `pattern_matcher=False` | 仍可能删除无用节点、调整推理图顺序、运行用户 hook、排序、更新 fake tensor 属性、恢复安全的原地写入并重新生成 `forward` | 合法 post 单图 |
| pattern 结构存在但附加条件不满足 | 该候选不应用，原节点继续留在图中 | 交给后续 pass 或 Lowering |
| pass 本身抛出异常或阶段不变量无法满足 | 本阶段没有把错误图伪装成成功出口 | 编译失败并向上报告 |

post 的 `reorder_for_locality` 只有 `is_inference=True` 且配置开启才运行；训练的前向图和反向图不走这个分支。更新完 fake tensor 属性后，编译器才恢复安全的原地写入，并把自动函数化包装展开成 clone 与底层写入；这两步会再次出现“覆盖已有 buffer”的节点，所以必须等前面那些按纯函数图分析的规则结束。

因此，“关闭 pattern matcher”不等于“完全跳过该阶段”，“没有匹配”也不等于编译失败。判断一张图为何保持原样，要指出具体被关闭或未满足的是哪组变换，而不是只看最终节点数量。

## 本层出口

经过这三个处理阶段后，交给 `GraphLowering` 的单图具有 post-grad 阶段承诺的表示与 meta。要判断某个具体 pattern 是否发生，必须观察对应阶段前后的 FX 图和匹配条件；只看最终数值相等无法证明命中哪一个 pass。

迁移题：一个改写需要知道 backward 是否使用前向中间值，同时又想依据最终 kernel 的 loop range 决定是否应用。前半需要 joint 阶段，后半直到 Lowering/Scheduler 才存在，不能由一个普通 FX pass 同时可靠拥有；应重新划分优化职责或传递明确的代价信息。

再看三个变化：

1. 一个规则需要实际 example inputs 才能运行。它即使适合 pre 外图，也不能假定递归子图同样收到 inputs；应支持 `None`、另建可靠传播机制，或换到具备所需 meta 的阶段。
2. 一个规则只想在 inference 中调整独立节点的局部顺序。post 有明确的 `is_inference` 分支；若放进 joint，训练图也会执行它，影响范围就超出了原本要求。
3. 一个规则把函数式 add 重新改成 in-place。若它早于依赖 functional graph 的 post patterns，就会破坏共同前提；当前顺序把 reinplace 放在这些工作之后。
