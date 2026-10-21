# `stable_topological_sort`：让每个 FX 节点回到全部输入之后

[返回图不变量](../README.md) · [返回图优化](../../README.md) · [返回总览](../../../README.md)

图改写可以先创建新节点、再重接 users，过程中暂时出现“消费者排在生产者前面”。FX 的线性节点表若保持这种顺序，生成 Python 或继续遍历都会先看到一个尚未定义的名字。

本叶子完整讲解当前 `/work/pytorch/torch/_inductor/pattern_matcher.py` 中连续的 `_args` 与 `stable_topological_sort`。它们不重新计算数值，也不创建节点；任务是移动现有 Node，使每条输入依赖都指向线性顺序中更早的节点，并尽量保留不受依赖约束的原顺序。

## 先用一张坏图建立问题

FX 节点当前顺序为：

```text
y, add, x
```

但 `add.args=(y,x)`。若按线性顺序生成代码，会先写 `add(y,x)`，之后才定义 x。`graph.lint()` 的实际错误是：

```text
Argument 'x' of Node 'add' was used before it has been defined
```

正确顺序不必强制成 x,y,add；原来 y 已在前面且与 x 无相互依赖，所以稳定修复得到 `y,x,add`。

## 声明范围的完整源码

先固定类型与状态。`List[...]` 是 Python 列表类型标注；`_args(add)` 在本例返回 `[y,x]`。进入 `stable_topological_sort` 时，`graph.nodes` 的线性顺序为 `[y,add,x]`；`pending` 反向保存这三项，`ready` 起初为空，`waiting` 也为空。遍历到 add 时 x 尚未 ready，所以 `waiting[x]=[add]`；x 就绪后再唤醒 add。代码结束时顺序为 `[y,x,add]`，`waiting={}`。

下面保留两个函数的全部产品语句、注释和原顺序。仅 `# 教学：` 开头的注释由教程加入。

```python
def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
    args: List[torch.fx.node.Argument] = []
    # 教学：递归走过位置参数和关键字参数的容器，把其中每个 Node 引用依次加入列表。
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    # 教学：列表先反转，pop() 因而从当前图的第一个节点开始取。
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = set()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        # 教学：每次唤醒都重新检查全部输入；重复引用也会出现在列表中。
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            # 教学：暂不移动 node，把它挂到当前列表最后一个未就绪输入名下。
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            # 教学：node 若不在 cursor 紧后方，就移动到 cursor 后；Node 身份不变。
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            # 教学：反转等待列表后压入 pending，使 pop() 恢复它们原来的先后顺序。
            pending.extend(reversed(waiting.pop(node, ())))

    # 教学：仍有等待项或 ready 数不等于图节点数，说明依赖无法由本图全部满足。
    assert not waiting and len(ready) == len(graph.nodes)
```

## `_args` 收集的是 Node 引用，不是普通常量

实验节点具有嵌套参数：

```python
args   = (x, [y, x])
kwargs = {"info": {"again": y}}
```

`torch.fx.map_arg` 递归穿过 tuple、list 和 dict，只对其中的 FX Node 调用 `args.append`；实际结果是：

```text
x, y, x, y
```

重复引用不会去重。普通整数、字符串和 None 不会进入依赖列表。排序只关心 Node→Node 的数据边；重复项最多使 `waiting_for` 重复，不会让一个尚未 ready 的依赖被误判为 ready。

## 三个集合与 cursor 各自保存什么

循环中每个图节点处于以下三种处理状态之一：

| 状态 | 容器 | 含义 |
| --- | --- | --- |
| 尚待首次检查或被重新唤醒 | `pending` | 末端用 `pop()` 取得下一项 |
| 已经放入合法前缀 | `ready` | 它的所有输入都更早 ready |
| 检查过但仍缺输入 | `waiting[dependency]` | 等 dependency ready 后重新压回 pending |

`cursor` 指向合法前缀的最后节点。`ready` 回答依赖是否完成；`cursor` 回答下一个完成节点应该放在哪里。节点即使已经处于正确的紧邻位置，也要加入 ready 并更新 cursor，只是不用执行 `append` 移动。

## 完整推演 `y,add,x`

初始：

```text
graph   = [y, add, x]
pending = [x, add, y]   # 右端 pop
ready   = {}
waiting = {}
cursor  = None
```

| 轮次 | pop | `waiting_for` | 动作 | 轮末关键状态 |
| ---: | --- | --- | --- | --- |
| 1 | y | `[]` | y ready | `ready={y}`, cursor=y |
| 2 | add | `[x]` | 挂到 waiting[x] | `waiting={x:[add]}` |
| 3 | x | `[]` | 把 x 移到 y 后；唤醒 add | graph=`[y,x,add]`，pending=`[add]` |
| 4 | add | `[]` | add 已紧跟 x，无需移动 | `ready={y,x,add}` |

循环结束时 waiting 为空，ready 数等于图中 3 个节点，断言通过。`lint()` 随后也实际通过。

## “最后一个未就绪输入”只是减少重查的优化

产品注释说：对已经排好或接近排好的图，挂到 `waiting_for[-1]` 可以只重查一次。但算法的正确性不依赖“最后一个参数一定最后 ready”。每次唤醒都会重新计算完整 `waiting_for`。

故意构造：

```text
线性顺序：add, y, x
add.args： (x, y)
```

具体状态：

1. 首次检查 add，缺 `[x,y]`，先挂到最后一项 y。
2. y ready 后唤醒 add；此时 x 仍未 ready，重新检查得到 `[x]`，改挂 x。
3. x ready 后再次唤醒，第三次检查 add 才进入 ready。

实际记录为 `y,x,add checks=3`。所以只等待一个 dependency 不会跳过其他依赖；不利顺序只会带来再次挂起。

## 等同一个依赖的多个消费者如何保持顺序

初始线性顺序是：

```text
neg(x), relu(x), x
```

neg 与 relu 依次加入 `waiting[x]=[neg,relu]`。x ready 时执行：

```python
pending.extend(reversed([neg, relu]))
```

pending 末端随后先 pop neg、再 pop relu，最终为：

```text
x, neg, relu
```

若不 reversed，栈式 pop 会把两个无相互依赖的消费者颠倒。这里的稳定性指：在依赖允许的范围内保留它们原来的相对顺序。

## 一条完全倒置的依赖链

实验先建立线性顺序：

```text
output, mul, add, x, y
```

依赖为：

```text
x,y → add → mul → output
```

状态传播如下：

```text
output 等 mul
mul    等 add
add    等最后缺失输入 y
x ready
y ready → 唤醒 add
add ready → 唤醒 mul
mul ready → 唤醒 output
```

最终实际顺序为 `x,y,add,mul,output`，并通过 lint。普通常量 2 不在 `_args(mul)` 的 Node 列表里，所以不会形成等待项。

## 已合法图为什么不被任意重排

测试图原顺序为 `b,a,add(a,b)`。b、a 都没有依赖，依次进入 ready；add 随后就绪。每次 `cursor.next is node`，没有任何移动，最终仍是 `b,a,add`，而不是按名字、参数位置或创建类型改成 a,b,add。

算法因此是稳定拓扑修复，不是唯一规范顺序。多种合法拓扑序中，它尽量保留输入的线性次序。

## 空图、外部依赖和环

### 空图

pending、ready、waiting 都为空，while 不执行；最终 `0==len(graph.nodes)`，正常返回。实验实际得到 `empty_graph=ok`。

### 引用了另一张 Graph 的 Node

若本图唯一消费者引用外部图的 placeholder，该外部 Node 会进入 waiting_for，却永远不会出现在本图 pending 中。循环结束后 waiting 非空，最终断言失败。实验实际得到 `external_dependency=AssertionError`。

### 环

若 a 依赖 b、b 又依赖 a，两者都进入 waiting，没有任何节点先 ready 来触发唤醒，最终同样断言失败。实际得到 `cycle=AssertionError`。

断言只能表明依赖未全部解决，不提供专门的环诊断，也不回滚循环中已做的移动。FX 正常图应当是同一 graph 所有依赖可满足的 DAG；这些是非法输入边界。

## 本函数与 meta、lint、recompile、DCE 的边界

stable sort 移动同一批 Node 对象，不调用节点 target。因此：

- 不重算 `node.meta['val']`；FakeTensorUpdater 属于 Q132。
- 不更新 GraphModule 已生成的 Python `forward`；修改 target/args 后仍需 `gm.recompile()`。
- 不完成所有结构验证；排序后用 `graph.lint()` 检查拓扑、所有权等不变量。
- 不删除无用户节点；`eliminate_dead_code()` 另行根据纯度与副作用处理。

实验把 add target 改成 mul 后，在 recompile 前调用仍得 5，recompile 后才得 6；这与排序无关。另一个 DCE 实验删除未使用的纯 sin，却保留有副作用的 copy_。这些对照防止把四项收尾误写成一个函数的效果。

## 改一个条件，自己推演

若顺序为 `consumer1,consumer2,y,x`，两者的 args 都是 `(x,y)`：它们先共同挂到 y；y 唤醒时各自发现 x 尚缺，再按原先顺序共同挂到 x；x ready 后仍以 consumer1、consumer2 的顺序进入合法前缀。最终是 `y,x,consumer1,consumer2`。

若把 consumer2 改为依赖 consumer1，它在第二次检查时即使 x、y 都 ready，也还会因 consumer1 未 ready 而继续等待；只有 consumer1 完成才被再次唤醒。决定顺序的是实际 Node 引用，不是第一次选择的 waiting key。

至此，`_args` 的容器遍历、pending/ready/waiting/cursor、移动和唤醒、重复重查、稳定顺序以及两个断言条件都有具体状态支撑。
