# 递归运行图优化：父图怎样把同一阶段送进控制流子图

[返回三个处理阶段](../README.md) · [返回图优化](../../README.md) · [返回总览](../../../README.md)

这一叶子解决一个很窄、却不能省略的问题：父 FX `GraphModule` 里的 `cond` 和 `while_loop` 节点引用了另外几张 `GraphModule`。若只对父图调用一次 pass，分支图和循环图就不会经历同一阶段。

这里完整讲解当前版本 `/work/pytorch/torch/_inductor/compile_fx.py` 的四个连续函数：一个子图名字枚举器，以及 pre、joint、post 三个递归入口。读完后，你应能拿一棵嵌套控制流图，写出各图被访问的次序、每层收到的额外参数，以及哪一步会替换父模块中的子图对象。

它们不是由同一个调用点连续调用。当前 `compile_fx.py` 中可以直接定位到这些 caller：

- `compile_fx(...)` 在把 Dynamo 交来的 `GraphModule` 交给 AOTAutograd 前调用 pre 入口；
- `fw_compiler_freezing(...)` 处理特殊的 freezing 推理路径，`compile_fx(...)` 内部的
  `_fw_compiler_base(...)` 处理非 freezing inference，两者分别调用 joint 入口；训练时，
  同样定义在 `compile_fx(...)` 内的 `partition_fn(...)` 在切分 forward/backward 前调用
  joint 入口；
- `_compile_fx_inner(...)` 收到一张即将单独编译的图后调用 post 入口；这张图可以是
  推理图，也可以是训练产生的 forward 或 backward 图。

本文只展开四个递归包装器怎样覆盖子图；pre/joint/post 内部各自改什么，仍由父篇和
对应 pass 实现负责。

## 先看这四个函数收到什么

普通 FX 节点通过输入边引用前一节点的结果。高阶控制流节点还会引用子 `GraphModule`：

```text
cond(pred, true_graph, false_graph, operands)
           ^^^^^^^^^^  ^^^^^^^^^^^
           父模块属性中的两张子图

while_loop(cond_graph, body_graph, carried_inputs, additional_inputs)
           ^^^^^^^^^^  ^^^^^^^^^^
           父模块属性中的两张子图
```

阶段函数一次只接收一张 `GraphModule`，默认遍历的也是这一个对象的 `graph`。分支图和循环图没有展开成父 `graph` 里的普通计算节点，而是保存在父模块属性中，所以必须显式取出并递归处理。

以 true 分支为例，当前实现接过下面四层对象：

```text
cond_node.args[1]
    = 一项 get_attr 节点
    ├─ name: "true_graph"       本实现读这个字符串
    └─ target: "true_graph"     该节点读取的模块属性路径

getattr(gm, "true_graph")
    = 真正的 true 分支 GraphModule
```

在本文实验以及当前编译流程交来的这类控制流图中，子图属性使用 `true_graph`、`false_graph` 等简单名字，`get_attr` 节点的 `name` 与该属性路径相同；产品代码没有另做检查，而是直接读取 `.name` 并交给 `getattr`。因此这是当前函数要求输入满足的前提，不是它已经验证过的全局合同，也不是在宣称任意 FX `get_attr` 节点的 `name` 永远等于 `target`。

名字枚举器只查找 target 精确等于 `torch.ops.higher_order.cond` 或
`torch.ops.higher_order.while_loop` 的节点。其他 HigherOrderOperator 即使也携带子图，
也不会自动进入这四个函数的递归范围。

## 声明范围的完整源码

阅读代码前固定主案例的输入状态：

| 名字 | 进入函数时的具体值或类型 | 后面哪里读取 |
| --- | --- | --- |
| `gm` | 父 `GraphModule`，其 graph 含一项 cond，参数 1、2 是 true/false 子图的 `get_attr` 节点 | `_get_subgraph_names` 调用 `gm.graph.find_nodes(...)` 找 cond/while 节点 |
| `find_nodes(...)` 的结果 | 本例是只含该 cond 节点的可迭代结果；while 查询为空 | `itertools.chain` 合并后由 `sorted` 按 FX 顺序遍历 |
| `true_subgraph_name` / `false_subgraph_name` | 两个 `get_attr` 节点的 `.name`，例如 `true_graph_0`、`false_graph_0`；在本实现接收的图中也是父模块子图属性名 | `getattr(gm, name)` 取出真正的子 `GraphModule` |
| 根层 `example_inputs` | 调用 pre 入口时传入的样例输入 | 只供根图的 `pre_grad_passes` 使用；递归子图固定收到 `None` |
| `is_inference` | 调用 post 入口时传入的布尔值 | 原样传给每一层子图和本层 `post_grad_passes` |

这段代码的可检查结果是：先分别处理 true、false 子图，再处理父图。只有 pre 包装器
接住递归返回的新 `GraphModule` 并替换父模块属性；joint/post 包装器不读取阶段函数的
返回值，依赖被调用函数修改原对象。

下面保留四个函数的全部产品语句、空行和原顺序。仅 `# 教学：` 开头的注释是本教程加入的；产品原注释原样保留。

```python
def _get_subgraph_names(gm):
    # 教学：分别查出当前图的 cond 与 while_loop 调用，再按 FX 节点顺序合并排序。
    for node in sorted(
        itertools.chain(
            gm.graph.find_nodes(op="call_function", target=torch.ops.higher_order.cond),
            gm.graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.while_loop
            ),
        )
    ):
        # 教学：cond 的第 1、2 个参数是指向 true/false 子图的 get_attr 节点。
        if node.target == torch.ops.higher_order.cond:
            true_subgraph_name = node.args[1].name
            false_subgraph_name = node.args[2].name
            yield true_subgraph_name
            yield false_subgraph_name
        # 教学：while_loop 的第 0、1 个参数是指向条件图和循环体图的 get_attr 节点。
        elif node.target == torch.ops.higher_order.while_loop:
            cond_subgraph_name = node.args[0].name
            body_subgraph_name = node.args[1].name
            yield cond_subgraph_name
            yield body_subgraph_name


def _recursive_pre_grad_passes(gm, example_inputs):
    # 教学：逐个取得当前图直接引用的子图；没有控制流时循环执行零次。
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        # as we don't have recursive example inputs, passing None here
        # 教学：先完成该子图及其后代；只有最外层保留调用方给的 example_inputs。
        new_subgraph = _recursive_pre_grad_passes(subgraph, example_inputs=None)
        # 教学：递归调用可能交回另一张 GraphModule，必须覆盖父模块的对应属性。
        setattr(gm, subgraph_name, new_subgraph)
    # 教学：所有直接子图完成后才处理当前 gm，并把可能的新 gm 交给调用者。
    return pre_grad_passes(gm, example_inputs)


def _recursive_joint_graph_passes(gm):
    # 教学：joint 同样先递归子图，但阶段函数的当前合同是原地修改。
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        _recursive_joint_graph_passes(subgraph)
    joint_graph_passes(gm)


def _recursive_post_grad_passes(gm, is_inference: bool = False):
    # 教学：is_inference 不在递归途中重算；同一个布尔值传到每一层。
    for subgraph_name in _get_subgraph_names(gm):
        subgraph = getattr(gm, subgraph_name)
        _recursive_post_grad_passes(subgraph, is_inference)
    post_grad_passes(gm, is_inference)
```

这里的 `_get_subgraph_names(gm)` 是生成器，不会先建立完整名字列表。外层 `for` 请求一个名字时，它运行到一次 `yield` 后暂停；外层可以递归处理整棵子树并写回，下一次循环再让生成器从暂停处继续。一个 cond 的 pre 路径因此实际交替执行：

```text
生成器 yield "true_graph" 后暂停
    → 外层取得 true 子图
    → 递归完成 true 的所有后代与 true 本身
    → setattr 写回 new_true
外层请求下一个名字，生成器从上次 yield 后恢复
    → yield "false_graph" 后再次暂停
    → 外层递归完成 false 子树并写回 new_false
生成器结束
    → 对父 gm 执行 pre_grad_passes
```

所以访问顺序不是先把所有名字“运行”一遍再统一递归。生成器的暂停点把每个名字与紧随其后的整棵子树处理连接起来。

## 第一条路径：普通图没有子图

若图只有 `sin`、`add_` 等普通节点，两个 `find_nodes` 都返回空集合，生成器不产生名字。三个递归入口各自直接执行当前图的阶段函数：

| 入口 | 循环次数 | 调用阶段函数 | 包装器的出口 |
| --- | ---: | --- | --- |
| pre | 0 | `pre_grad_passes(gm, example_inputs)` | 返回该调用的结果 |
| joint | 0 | `joint_graph_passes(gm)` | 隐式返回 `None`，变化留在 `gm` |
| post | 0 | `post_grad_passes(gm, is_inference)` | 隐式返回 `None`，变化留在 `gm` |

实验中的请求是：

```python
v = x.view(-1)
old = torch.sin(v)
x.add_(1)
later = torch.cos(v)
return old, later, x
```

pre hook 看到 `view + sin + add_ + cos`；AOTAutograd functionalization 之后，joint hook
看到函数式 `add` 和显式 `copy_`，而 post hook 还把两项 `view` 规范化为 `reshape`。
输入 `[0,1,2]` 执行后，`old` 保存写入前的正弦值，`later` 通过共享 storage 的 view
读到写入后的 `[1,2,3]`，调用者的 `x` 也变为 `[1,2,3]`。这个观测说明三个入口确实
位于不同阶段。它本身没有控制流，因此只覆盖上表的零次循环路径。

## 第二条路径：一个 cond 怎样被处理

设父图含一个 cond，属性名为 `true_graph`、`false_graph`。pre 入口的具体状态依次是：

| 时刻 | `subgraph_name` | `subgraph` | 动作与结果 |
| --- | --- | --- | --- |
| 1 | `true_graph` | true 分支 GraphModule | 递归调用，子层 `example_inputs=None`；返回 `new_true` |
| 2 | `true_graph` | 原 true 分支 | `setattr` 将父模块属性改成 `new_true` |
| 3 | `false_graph` | false 分支 GraphModule | 同样递归并写回 `new_false` |
| 4 | 循环结束 | 父 `gm` | 对已持有两张新子图的父图调用 `pre_grad_passes` |

所以这是后序遍历：保持直接子图从左到右的次序，但把当前图放到所有子图之后。若 pre pass 返回另一张 `GraphModule`，替换不会凭空传播；正是 `setattr` 把递归结果接回父模块，而最外层调用者还必须接住整个函数的返回值。

用对象身份看区别更直接。进入 true 分支前，令 `old_true = gm.true_graph`：

```text
pre：  递归返回 new_true
      setattr 后 gm.true_graph is new_true 为 True
      gm.true_graph is old_true 为 False

joint/post：阶段函数修改 old_true 本身
            调用后 gm.true_graph is old_true 仍为 True
```

joint 和 post 的次序相同，但没有 `new_subgraph` 与 `setattr`。当前包装器根本不接它们
的返回值，因此它们依赖阶段函数原地修改传入图的合同。当前
`joint_graph_passes(gm)` 修改 `gm` 后还会返回同一个对象，但这个递归包装器忽略该返回值；
`post_grad_passes(gm, is_inference)` 则没有显式返回。若其中任一阶段以后只返回新图而不
再修改原对象，包装器就必须像 pre 一样接住并写回。这里能断言的是对象上的修改会留下；
具体修改 graph、meta 还是其他状态，由被调用阶段函数决定。

## 第三条路径：while、cond 与嵌套图的完整次序

独立实验建立了这棵只用于观察递归的图树：

```text
root
├─ while_cond
│  └─ cond
│     ├─ wc_true
│     └─ wc_false
├─ while_body
├─ true_graph
│  └─ cond
│     ├─ t_true
│     └─ t_false
└─ false_graph
```

根图中的 `while_loop` 节点写在 `cond` 节点之前。`_get_subgraph_names(root)` 的实际结果是：

```text
cond 查询结果：  [root_cond]
while 查询结果： [root_while]
chain 后：        [root_cond, root_while]
按 FX 位置 sorted：[root_while, root_cond]

root_while 依次 yield：while_cond, while_body
root_cond 依次 yield： true_graph, false_graph

最终直接名字：while_cond, while_body, true_graph, false_graph
```

继续递归后，三个入口都得到同一后序访问序列：

```text
wc_true → wc_false → while_cond → while_body
        → t_true → t_false → true_graph → false_graph → root
```

逐步看前半段：先从 root 取 `while_cond`；它内部又有 cond，于是先处理 `wc_true`、`wc_false`；两支结束才处理 `while_cond`；返回 root 的循环后，再处理 `while_body`。随后才进入根图的 true/false 两支。

`sorted(...)` 稳定的是编译期遍历 FX 节点的顺序。它没有执行 cond，也没有重复执行 while body；运行时选哪条分支、循环多少轮，仍由高阶算子的语义决定。

## pre、joint、post 的状态差异

实验保留产品递归包装器，只把三个阶段函数换成可记录状态的小函数，得到：

| 可观察状态 | pre | joint | post |
| --- | --- | --- | --- |
| 后序访问顺序 | 上述 9 张图 | 相同 | 相同 |
| 根层额外参数 | 原 `example_inputs` 对象 | 无 | `is_inference=True` |
| 子层额外参数 | 全部为 `None` | 无 | 每层均为 `True` |
| 阶段函数返回另一张图 | 会由父层 `setattr` 接住 | 包装器不接返回值 | 包装器不接返回值 |
| 包装器返回 | 新根图 | `None` | `None` |
| 本例模块身份 | 根与四个直接子图均替换 | 调用前后全部相同 | 调用前后全部相同 |

“子层 pre 收到 `None`”不表示子图没有输入。它只表示这个递归入口没有一份与每张子图参数一一对应的样例输入列表。子图 pass 若需要形状或类型，应使用当前接口真正能取得的资料，不能假定这里总会传来运行样例。

根函数的输入列表也不能直接原样当作每张子图的样例：cond 分支收到的是 `operands`
对应的参数，while 的条件图和循环体还会接收 carried values 等各自签名。当前包装器
没有重建这些逐子图样例，源码在递归 pre 调用处直接传 `None`。

## 用四个变化检查是否真正理解

第一，若删除根图的 while，只保留那个嵌套 cond，访问序列变为：

```text
t_true → t_false → true_graph → false_graph → root
```

理由不是“cond 总在后面”，而是根图只剩 `true_graph`、`false_graph` 两个直接子图，其中 true 分支仍要先完成自己的两个后代。

第二，若 pre 的 true 分支阶段函数返回新对象，但父层漏掉 `setattr`，后续处理父图时仍会从属性取得旧 true 分支；递归产生的新对象将失去连接。joint/post 不需要这次写回，是因为当前包装合同采用原地修改，而不是因为控制流子图不重要。

再检查两个变化：若在第一次 `yield "true_graph"` 暂停时 true 子图内部还有一个 cond，生成器何时才会恢复并产生 `false_graph`？答案是内部 cond 的两支、true 子图本身以及 `new_true` 写回全部完成之后。若 joint 阶段函数以后改为返回新 `GraphModule`，包装器也必须保存递归返回值并 `setattr`；否则新对象会像缺少 pre 写回时一样失去连接。

至此，本篇范围内的循环、条件分支、参数传播和正常出口都已落到具体状态。三个阶段及其
控制流子图现在都会收到相应 pass。下一篇进入
[Pattern Matcher 总览](../../02-pattern-matching/README.md)：先看一条局部改写规则怎样
成为 `PatternEntry`、注册进 `PatternMatcherPass`，以及阶段入口怎样调用它；建立这些
对象关系后，再打开候选扫描与应用的源码。
