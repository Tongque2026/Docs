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

一次安全的图变换依次完成三件事：

```text
决定规则在哪种图上运行
        ↓
找到目标节点关系，建立替代节点并重接使用者
        ↓
同步节点顺序、结果属性和可执行 forward，检查引用
        ↓
GraphLowering 读取整理后的 GraphModule
```

下面先让同一条规则走完这条线，后文再说明三项职责分别把什么交给下一项。

## 同一条规则怎样从安排走到 Lowering

独立实验为当前 `PatternMatcherPass` 注册了下面的规则：

```text
匹配：mul(add(x,y), 2)
替换：add(mul(x,2), mul(y,2))
条件：x 的 dtype 是 float32
```

输入 `x=[1,2,3]`、`y=[10,20,30]` 时，两种写法都得到 `[22,44,66]`。

这条规则只需要规范化的 `add`、`mul` 节点和 dtype，所以可以在拆分后的单图上运行。编译器把调用它的 pass 放在 post-grad。pass 收到实际图后，Pattern Matcher 先确认 `mul` 的输入来自 `add`、常数为 2，并从节点属性读出 float32 条件已经满足。

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

节点和引用改完后，图还要恢复为下一阶段可以直接读取的状态：

| 需要同步的状态 | 修改后要达到的结果 |
| --- | --- |
| 节点顺序 | 产生结果的节点排在读取结果的节点之前 |
| 节点属性 | 新节点的 shape、dtype、device、stride 对应新计算 |
| 可执行 `forward` | `recompile()` 让 Python 代码反映新节点图 |
| 图结构 | `lint()` 检查先用后定义、错误所属图和悬空引用等问题 |

这些状态同步完成后，GraphLowering 收到的是最终图。实验中的 float32 图记录 `matches=1`，新图和原函数结果相同。输入改成 int64 后，dtype 条件不满足，`matches=0`；这项规则不改图，原节点继续进入后续 pass，最终仍可交给 Lowering。

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
    old = torch.sin(x)
    x.add_(1)
    return old, x
```

输入 `x=[0,1,2]` 时，`old` 必须读取修改前的 x，调用者持有的 x 最终变成 `[1,2,3]`。独立实验 [`experiments/16_pass_stage_graphs.py`](../experiments/16_pass_stage_graphs.py) 观察到：

```text
pre-grad： old=sin(x)；随后直接出现 x.add_(1)
joint：    old=sin(x)；new=add(x,1)；copy_(x,new)
post-grad：本例仍是 sin → add → copy_
```

需要识别用户是否写了 `add_` 的规则要在 pre-grad 取得信息；只处理函数式 `add` 和显式 `copy_` 的规则可以在后面的图上运行。阶段编排负责选择和依次调用这些 pass，并把一项 pass 的输出交给下一项。详细过程见 [阶段编排](01-pass-stages/README.md)。

## Pattern Matcher 怎样改变节点和引用

Pattern Matcher 接收当前图和已注册规则，先从可能的末端节点寻找候选，再沿节点参数核对上游结构，并检查 dtype、device、shape 等附加条件。匹配成功后，它把 replacement 图中的新节点复制到当前图，把旧结果的外部使用者改接到新结果，最后删除已经没有使用者的旧节点。

它交出的核心结果就是新的节点关系。上面的 float32 案例中，Lowering 将读到两个 `mul` 和一个新 `add`；int64 案例中，它仍读到原来的 `add→mul`。候选查找、参数绑定和 replacement 嵌入见 [Pattern Matcher](02-pattern-matching/README.md)。

## 图状态怎样跟上节点变化

节点图、节点属性和 `GraphModule.forward` 是同一份程序的不同记录。修改其中一份后，其余记录必须跟着更新。

独立实验 [`experiments/03_graph_invariants.py`](../experiments/03_graph_invariants.py) 把 x 故意移动到使用它的 add 后面：

```text
错误顺序：y, add(y,x), x
lint：add 使用了尚未定义的 x
稳定拓扑排序后：y, x, add(y,x)
```

排序只改变节点排列，不改变 `add` 读取 y、x 的关系。同一实验还把 `a+b` 的节点 target 改成乘法：调用 `recompile()` 前，`gm(2,3)` 仍执行旧 `forward`，结果是 5；重新生成 `forward` 后结果才变成 6。若替换改变结果属性，fake tensor 传播还要为新节点产生对应的 shape、dtype、device 和 stride。

稳定排序、属性更新、`recompile()` 与 `lint()` 共同交出一张内部记录一致的图。具体算法见 [图状态维护](03-graph-invariants/README.md)。

## 图优化的出口

图优化改变的是 FX 节点和结果引用。逐元素 `add→relu→mul` 即使在 post-grad 出口仍是三个 FX 节点，Scheduler 以后也可能把它们组成一个生成单元，最终由一个 C++ 或 Triton kernel 完成。FX 图替换和 Scheduler 融合发生在不同对象上。

读完本篇，可以沿下面的顺序判断一项变化属于哪里：

1. 规则依赖哪种图表示，决定它在哪个编译位置运行；
2. matcher 是否改变了 FX 节点、结果引用和仍需保留的使用者；
3. 修改后的节点顺序、结果属性、`forward` 和引用是否合法；
4. GraphLowering 最终收到哪张图。

继续阅读：

1. [阶段编排：三个位置分别提供什么信息](01-pass-stages/README.md)
2. [Pattern Matcher：怎样完成一次局部替换](02-pattern-matching/README.md)
3. [图状态维护：怎样恢复一张可继续读取的图](03-graph-invariants/README.md)
