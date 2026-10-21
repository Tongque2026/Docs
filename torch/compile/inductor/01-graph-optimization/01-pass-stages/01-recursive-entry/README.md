# 递归运行图优化：父图怎样把同一阶段送进控制流子图

[返回三个处理阶段](../README.md) · [返回图优化](../../README.md) · [返回总览](../../../README.md)

这一叶子解决一个很窄、却不能省略的问题：父 FX `GraphModule` 里的 `cond` 和 `while_loop` 节点引用了另外几张 `GraphModule`。若只对父图调用一次 pass，分支图和循环图就不会经历同一阶段。

这里完整讲解当前版本 `/work/pytorch/torch/_inductor/compile_fx.py` 的四个连续函数：一个子图名字枚举器，以及 pre、joint、post 三个递归入口。读完后，你应能拿一棵嵌套控制流图，写出各图被访问的次序、每层收到的额外参数，以及哪一步会替换父模块中的子图对象。

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

在这段实现里，节点参数中的 `get_attr` 节点负责指出属性名。枚举器产生这些名字，递归入口再用 `getattr(gm, name)` 取到真正的子 `GraphModule`。

## 声明范围的完整源码

阅读代码前固定主案例的输入状态：

| 名字 | 进入函数时的具体值或类型 | 后面哪里读取 |
| --- | --- | --- |
| `gm` | 父 `GraphModule`，其 graph 含一项 cond，参数 1、2 是 true/false 子图的 `get_attr` 节点 | `_get_subgraph_names` 调用 `gm.graph.find_nodes(...)` 找 cond/while 节点 |
| `find_nodes(...)` 的结果 | 本例是只含该 cond 节点的可迭代结果；while 查询为空 | `itertools.chain` 合并后由 `sorted` 按 FX 顺序遍历 |
| `true_subgraph_name` / `false_subgraph_name` | 两个 `get_attr` 节点的名字，例如 `true_graph_0`、`false_graph_0` | `getattr(gm, name)` 取出真正的子 `GraphModule` |
| `example_inputs` 或阶段参数 | 调用当前 pre/joint/post 函数时一并传入的样例或配置 | 递归调用子图的同阶段函数时继续传递 |

这段代码的可检查结果是：先分别处理 true、false 子图，再处理或返回父图；若子图函数返回了新的 `GraphModule`，父模块对应属性会被替换。

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

## 第一条路径：普通图没有子图

若图只有 `sin`、`add_` 等普通节点，两个 `find_nodes` 都返回空集合，生成器不产生名字。三个递归入口各自直接执行当前图的阶段函数：

| 入口 | 循环次数 | 调用阶段函数 | 包装器的出口 |
| --- | ---: | --- | --- |
| pre | 0 | `pre_grad_passes(gm, example_inputs)` | 返回该调用的结果 |
| joint | 0 | `joint_graph_passes(gm)` | 隐式返回 `None`，变化留在 `gm` |
| post | 0 | `post_grad_passes(gm, is_inference)` | 隐式返回 `None`，变化留在 `gm` |

实验中的请求是：

```python
old = torch.sin(x)
x.add_(1)
return old, x
```

pre hook 仍看到 `sin + add_`；AOTAutograd functionalization 之后，joint 与 post hook 看到函数式 `add`，再用 `copy_` 把新值写回输入。这个观测说明三个入口确实位于不同阶段。它本身没有控制流，因此只覆盖上表的零次循环路径。

## 第二条路径：一个 cond 怎样被处理

设父图含一个 cond，属性名为 `true_graph`、`false_graph`。pre 入口的具体状态依次是：

| 时刻 | `subgraph_name` | `subgraph` | 动作与结果 |
| --- | --- | --- | --- |
| 1 | `true_graph` | true 分支 GraphModule | 递归调用，子层 `example_inputs=None`；返回 `new_true` |
| 2 | `true_graph` | 原 true 分支 | `setattr` 将父模块属性改成 `new_true` |
| 3 | `false_graph` | false 分支 GraphModule | 同样递归并写回 `new_false` |
| 4 | 循环结束 | 父 `gm` | 对已持有两张新子图的父图调用 `pre_grad_passes` |

所以这是后序遍历：后代先、当前图后。若 pre pass 返回另一张 `GraphModule`，替换不会凭空传播；正是 `setattr` 把递归结果接回父模块，而最外层调用者还必须接住整个函数的返回值。

joint 和 post 的次序相同，但没有 `new_subgraph` 与 `setattr`。它们依赖各自阶段函数原地修改传入的图。这里能断言的是对象上的修改会留下；具体修改 graph、meta 还是其他状态，由被调用阶段函数决定，递归包装器没有额外保证。

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
while_cond, while_body, true_graph, false_graph
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

## 用两个变化检查是否真正理解

第一，若删除根图的 while，只保留那个嵌套 cond，访问序列变为：

```text
t_true → t_false → true_graph → false_graph → root
```

理由不是“cond 总在后面”，而是根图只剩 `true_graph`、`false_graph` 两个直接子图，其中 true 分支仍要先完成自己的两个后代。

第二，若 pre 的 true 分支阶段函数返回新对象，但父层漏掉 `setattr`，后续处理父图时仍会从属性取得旧 true 分支；递归产生的新对象将失去连接。joint/post 不需要这次写回，是因为当前包装合同采用原地修改，而不是因为控制流子图不重要。

至此，四个函数的每个循环、条件分支、参数传播和出口都已落到具体状态。下一叶子进入 pattern matcher 的候选扫描与应用。
