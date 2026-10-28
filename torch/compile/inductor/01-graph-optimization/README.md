# 图优化：把整理后的 FX 图交给 Lowering

[返回总览](../README.md) · 下一篇：[Lowering](../02-lowering/README.md)

图优化收到一张仍能执行的 FX `GraphModule`，修改其中的节点和结果引用，再交出一张仍能执行的 `GraphModule`。Lowering 随后逐个读取这张图的节点，把它们变成 Inductor 内部的计算和存储表示。

先看一张最小图：

```text
x ─┐
   ├─ add(a) ─→ relu(b) ─→ output(b)
y ─┘
```

每个方框是一项输入、计算或输出。箭头表示后一个节点读取前一个节点的结果：`relu` 读取 `add` 的结果 `a`，`output` 读取 `relu` 的结果 `b`。节点旁还可保存结果的 shape、dtype、device 和 stride；`GraphModule` 则保存整张节点图以及据此生成的 Python `forward`。

下文的 matcher 局部替换依次经过三项交接：

```text
1. 阶段编排：决定规则在哪种图上运行，并调用相应 pass
        ↓
2. Pattern Matcher：保存规则；被调用后寻找目标关系、建立替代节点并重接使用者
        ↓
3. 图状态维护：同步节点顺序、结果属性和可执行 forward，检查引用
        ↓
GraphLowering 读取整理后的 GraphModule
```

阶段编排还可以直接调用 DCE 等非 matcher pass；它们不经过“寻找 pattern 并建立替代节点”
这一步，改完节点后仍要进入相应的图状态维护，再把合法图交给 Lowering。因此三项是本篇
要理解的协作职责，不表示每一种 pass 都必须依次调用三个子模块。matcher 对象何时被阶段
调用属于第一项；对象中的规则表和一次 `apply` 属于第二项。

下面用直接调用 matcher 的实验观察同一条规则经过三项职责时的对象交接，后文再分别展开。

## 用同一条规则看三项职责怎样交接

图优化阶段通常要尝试许多局部改写规则。Inductor 用
`PatternMatcherPass` 对象保存一组这样的规则，并在阶段运行到它时，把当前 FX
图交给它检查。对象内部的规则按“末端节点调用什么操作”分组；这样它可以先从图中
找出可能命中的节点，再尝试对应规则。

一项具体规则、一组规则和阶段入口之间的关系是：

```text
一项局部改写规则
    │ 注册到
    ▼
某个 PatternMatcherPass 对象中的规则表
    │ 某一阶段的代码调用 apply(当前图)
    ▼
在当前 FX 图中尝试这组规则
    │
    ├─ 命中：直接修改图中的节点和引用
    └─ 未命中：保留原节点关系
```

图优化运行在编译流程的三个位置：Dynamo 之后、AOTAutograd 拆分训练图之前，以及
每张前向图或反向图进入 Lowering 之前。后文分别称它们为 pre-grad、joint 和
post-grad。一个位置可以依次调用多个 `PatternMatcherPass` 对象，每个对象也可以
保存多条规则；一个这样的对象不等于整个处理阶段。

例如 post-grad 模块在加载时建立一个含三个 `PatternMatcherPass` 对象的
`pass_patterns` 列表。开启 `config.pattern_matcher` 后，首次运行还会通过 `lazy_init()`
导入相应注册模块，让延迟注册的规则进入这三个对象；`post_grad_passes` 随后遍历同一个
列表，对三个对象依次调用 `apply(gm.graph)`。后续图继续使用这批已经建立并注册规则的
matcher 对象，不会为每张图重建规则表。配置关闭时不运行这组三个 matcher。其他阶段也
可以保存并调用自己的 matcher 对象，具体容器在阶段编排子篇展开。

阶段还可以安排不依赖 matcher 的图处理。例如当前 post-grad 在运行 pattern 前，若
`config.dce` 开启，会先删除没有使用者且没有必须保留副作用的死节点。这个 DCE 是阶段
编排中的独立动作，不是 replacement helper 删除本次匹配旧节点的那一步；pre-grad 和
joint 也不共享这一条固定的 post-grad 入口流程。

下面的独立实验先创建一个名为 `tutorial_distribute_mul` 的
`PatternMatcherPass` 对象，再建立一条规则记录并把它加入这个对象的规则表。这条记录
包含三部分：

| 规则记录中的内容 | 本例的具体值 |
| --- | --- |
| 要匹配的节点关系 | `mul(add(x,y), 2)` |
| 结构命中后的附加检查 | 捕获到的 x 的 dtype 是否为 float32 |
| 检查通过后执行的改图动作 | 建立 `add(mul(x,2), mul(y,2))` 并接替旧 mul 的结果 |

注册动作的输入是这三项内容和目标 matcher 对象；结果是 matcher 的规则表多出这条记录。
运行阶段随后从同一个对象中取得它。本篇先追踪这条记录怎样影响当前图；模式对象和注册
装饰器怎样构造这条记录，在 Pattern Matcher 子篇展开。注册装饰器还会把原处理函数原样
返回，使后续 Python 定义仍引用这个函数；图优化依赖的是它已经把规则写入 matcher 的副作用。

输入 `x=[1,2,3]`、`y=[10,20,30]` 时，两种写法都得到 `[22,44,66]`。

这条规则只需要规范化的 `add`、`mul` 节点和 dtype，所以可以在拆分后的单图上运行。
这条教学规则没有写入产品的 `pass_patterns` 列表。实验在建立 `GraphModule gm` 后直接
调用 `matcher.apply(gm)`；`apply` 同时接受 `GraphModule` 或其中的 `Graph`，产品的
post-grad 入口传入 `gm.graph`。实验借此单独观察同一 matcher 接口。根据规则只依赖规范化
`add`、`mul` 和 dtype 这一前提，若产品需要采用这条规则，可以把它注册到 post-grad
会遍历的某个 matcher；本实验不声称已经完成这项产品接入。

实验把 `gm` 传给 `apply`，matcher 从中取得 `gm.graph`；实际 post-grad 入口直接把
`gm.graph` 传给 `apply`。两种调用最终修改的都是调用者持有的同一张图。
`apply` 先确认 `mul` 的输入来自 `add`、常数为 2，再把捕获到的 x 交给附加检查；检查从
x 的节点属性读出 dtype 为 float32。

在尝试这项结构和附加检查之前，matcher 还会按 mutation 划分图中的区域。一次匹配若需要
跨过写入边界，旧值和新值可能已不再代表同一状态，候选会在调用附加检查和规则动作之前
被拒绝。实验中的 `add → copy_ → mul` 虽然含有相似的 add/mul 形状，但横跨 mutation
region，`apply` 返回 0，图保持不变。

匹配成功后的节点变化是：

```text
改写前：x,y → add(a) → mul(a,2)(b) → output(b)

建立：  x → mul(m0) ─┐
                       ├→ add(d)
          y → mul(m1) ─┘

重接：output 原来读取 b，现在读取 d
删除：b 已无使用者；a 也已无其他使用者，因此删除

改写后：x → mul(m0) ─┐
                       ├→ add(d) → output(d)
          y → mul(m1) ─┘
```

如果 `a` 还直接连接第二个输出，旧 `mul(b)` 仍可删除，但旧 `add(a)` 必须保留：

```text
x,y → add(a) ─────────────→ output1(a)
       └─ 原 mul 被替换 ─→ output0(d)
```

替换只改由旧 `mul` 产生的结果；没有被替换接管的使用者继续读取旧 `add`。因此，决定旧节点能否删除的是它改写后还有没有使用者。

从外部看，`PatternMatcherPass.apply` 负责遍历候选并调用命中规则的改图动作。本例的
动作调用 replacement helper；helper 把替换计算加入当前图、让旧结果的使用者改读新
结果，并删除已经无用的旧节点。这些工作都发生在本次 `apply` 调用内部。helper 没有
另行返回新节点；它返回 `None`，通过原地修改 `gm.graph` 完成交接。
`apply` 返回时，传入的 `gm.graph` 已经改变，但它没有替整个阶段重新生成
`GraphModule.forward`。阶段代码仍持有原来的 `gm`，接着处理其中这张已经改变的图，
并在交给下一阶段前完成图状态维护：

| 需要同步的状态 | 修改后要达到的结果 |
| --- | --- |
| 节点顺序 | 产生结果的节点排在读取结果的节点之前 |
| 节点属性 | 新节点的 shape、dtype、device、stride 对应新计算 |
| 可执行 `forward` | `recompile()` 让 Python 代码反映新节点图 |
| 图结构 | `lint()` 检查先用后定义、错误所属图和悬空引用等问题 |

当前 post-grad 入口用 FakeTensor 形式的示例值增量更新受影响节点的属性；这个过程只计算
shape、dtype、device、stride 等属性，不读取真实数据。post-grad 随后还会整理节点顺序、
调用 `recompile()`，并用 `gm.graph.lint()` 检查结构。pre-grad 只在特定 predispatch 路径
运行 ShapeProp，joint 也没有使用同一套 FakeTensorUpdater；它们按各自出口需要执行排序、
检查和重新编译。因此表格描述的是改图后必须保持一致的记录，不表示三个阶段调用同一组
维护函数；具体顺序在相应子篇说明。

这些状态同步完成后，GraphLowering 收到的是最终图。实验中的 float32 图让 `apply`
返回 1，新图和原函数结果相同。输入改成 int64 后，dtype 条件不满足，`apply` 返回 0；这项
规则不改图，原节点继续进入后续 pass，最终仍可交给 Lowering。`apply` 的计数在候选通过
结构匹配、mutation-region 检查和附加检查、即将进入规则记录的 `entry.apply` 时增加。
本教学规则的 `entry.apply` 随后确实完成替换，所以返回 1 与一次改图同时发生；框架也允许
只观察而不改图的 handler，这时计数增加不保证节点关系改变。同一规则若有两个候选进入
`entry.apply`，次数增加两次；附加检查失败的候选不计数。

实验见 [`experiments/02_pattern_matcher.py`](../experiments/02_pattern_matcher.py)。

## 运行位置决定规则能看见什么

同一项用户操作经过 AOTAutograd 和训练图拆分后，会有不同的图表示：

```text
Dynamo 交出的图，其中还可能直接出现 x.add_(1)
  └─ pre-grad passes
       └─ AOTAutograd 将许多原地计算写成普通计算和显式写回
            └─ joint passes：训练时前向和反向仍在同一张图里
                 └─ partition 拆出分别编译的前向图和反向图
                      └─ 每张单图运行 post-grad passes
                           └─ GraphLowering
```

| 位置 | 此处能看见的信息 | 适合在此运行的规则 |
| --- | --- | --- |
| pre-grad | 用户写下的原地操作，以及多个 tensor 可能共享存储的关系 | 需要识别原始调用写法、mutation 或 alias 的规则 |
| joint | 训练的前向结果和使用它的反向计算仍在一张图中 | 需要同时比较前向保存和反向重算的规则 |
| post-grad | 拆分后的规范化前向图或反向图；许多原地计算已与显式写回分开 | 只依赖普通 ATen 节点、shape、dtype 等单图信息的规则 |

例如：

```python
def fn(x):
    v = x.view(-1)
    old = torch.sin(v)
    x.add_(1)
    later = torch.cos(v)
    return old, later, x
```

输入 `x=[0,1,2]` 时，`old` 必须读取修改前的 x，`later` 必须读取修改后的值，调用者
持有的 x 最终变成 `[1,2,3]`。独立实验
[`experiments/16_pass_stage_graphs.py`](../experiments/16_pass_stage_graphs.py) 观察到：

```text
pre-grad： old=sin(view(x))；x.add_(1)；later=cos(同一个 view)
joint：    old 读取 view(x)；new=add(x,1)；later 读取 view(new)；末尾 copy_(x,new)
post-grad：仍用旧值路径计算 old、新值路径计算 later，并在末尾写回 x
```

这里的 post-grad 文本采自 `post_grad_custom_post_pass` hook。该 hook 位于后续稳定排序、
FakeTensor 属性增量更新、reinplace、`recompile()` 和 `lint()` 之前，所以它证明的是这个
hook 时刻已经形成旧值路径、新值路径和显式写回；最终 post-grad 出口还要继续完成上述
维护。本实验同时核对最终编译结果，但不把 hook 文本当成维护完成后的最终图快照。

需要识别用户是否写了 `add_` 的规则要在 pre-grad 取得信息；后面的图已经用不同节点表示
旧值读取、新值读取和对调用者的最终写回。阶段编排负责选择和依次调用这些 pass，并把一项
pass 的输出交给下一项。详细过程见 [阶段编排](01-pass-stages/README.md)。

## Pattern Matcher 怎样改变节点和引用

阶段代码调用某个 `PatternMatcherPass.apply(graph)` 后，该对象读取自己的规则表，
先从可能的末端节点寻找候选，再沿节点参数核对上游结构，并检查 dtype、device、shape
等附加条件。匹配成功后，它把匹配结果交给规则记录中的改图动作。匹配结果保存本次找到
的旧 add、旧 mul，以及从图中捕获的 x、y 节点。本例的动作调用 replacement helper，
同时传入匹配结果、一张描述 `x*2+y*2` 的小型 FX 图和捕获的 x、y。helper 将小图的
placeholder 分别绑定到当前图中的 x、y，把计算节点复制进当前图，以新 add 接替旧 mul
的结果引用，最后删除已经没有使用者的旧节点。helper 返回 `None`；规则动作和后续阶段
看到结果，是因为同一张 `gm.graph` 已被原地修改。整个过程仍在 `apply` 调用内部完成。

`apply` 把传入的图交给通过检查的规则动作，返回值是进入 `entry.apply` 的候选次数。
对本教学替换规则，返回 1 时规则动作原地改了一处图；对只记录事件的 handler，计数也可
增加而图不变。无论哪种 handler，返回值都不是另一张新图。

它交出的核心结果就是新的节点关系。上面的 float32 案例中，Lowering 将读到两个 `mul` 和一个新 `add`；int64 案例中，它仍读到原来的 `add→mul`。候选查找、参数绑定和 replacement 嵌入见 [Pattern Matcher](02-pattern-matching/README.md)。

## 图状态怎样跟上节点变化

节点图、节点属性和 `GraphModule.forward` 是同一份程序的不同记录。修改其中一份后，其余记录必须跟着更新。

图状态维护接住 matcher 修改后的同一张图。独立实验
[`experiments/03_graph_invariants.py`](../experiments/03_graph_invariants.py) 把 x
故意移动到使用它的 add 后面：

```text
错误顺序：y, add(y,x), x
lint：add 使用了尚未定义的 x
稳定拓扑排序后：y, x, add(y,x)
```

排序只改变节点排列，不改变 `add` 读取 y、x 的关系。同一实验还把 `a+b` 的节点 target
改成乘法：调用 `recompile()` 前，`gm(2,3)` 仍执行旧 `forward`，结果是 5；重新生成
`forward` 后结果才变成 6。若替换改变结果属性，当前 post-grad 的属性更新步骤用不含真实
数据的 FakeTensor 示例值执行受影响节点，从而得到对应的 shape、dtype、device 和 stride。

在当前 post-grad 路径中，稳定排序、FakeTensor 属性增量更新、`recompile()` 与 `lint()`
共同交出一张内部记录一致的图。pre-grad 只在特定 predispatch 路径运行 ShapeProp，joint
也没有使用同一套 FakeTensorUpdater；它们按各自出口需要完成排序、检查和重新编译。因此，
三种位置都要交出合法图，但不调用完全相同的属性维护流程。具体算法见
[图状态维护](03-graph-invariants/README.md)。

## 图优化的出口

图优化改变的是 FX 节点和结果引用。如果 post-grad 没有规则改写某条逐元素
`add→relu→mul` 链，它交给 Lowering 时仍是三个 FX 节点；Scheduler 以后仍可能把它们组成
一个生成单元，最终由一个 C++ 或 Triton kernel 完成。FX 图替换和 Scheduler 融合发生在
不同对象上。

读完本篇，可以沿三项图优化职责和一个出口判断一项变化属于哪里：

1. 规则依赖哪种图表示，决定它在哪个编译位置运行；
2. matcher 是否改变了 FX 节点、结果引用和仍需保留的使用者；
3. 修改后的节点顺序、结果属性、`forward` 和引用是否合法；
出口：GraphLowering 最终收到哪张图。

继续阅读：

1. [阶段编排：三个位置分别提供什么信息](01-pass-stages/README.md)
2. [Pattern Matcher：怎样完成一次局部替换](02-pattern-matching/README.md)
3. [图状态维护：怎样恢复一张可继续读取的图](03-graph-invariants/README.md)
