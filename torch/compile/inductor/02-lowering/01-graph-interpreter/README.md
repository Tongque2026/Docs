# GraphLowering：一个 FX 节点怎样进入 IR 环境

[返回 Lowering](../README.md) · [返回总览](../../README.md)

实现叶子：[输入登记](01-node-entry-source/README.md)、[算子与常量分派](02-dispatch-source/README.md)、[输出与输入修改](03-output-source/README.md)、[`run_node` 外围处理](04-run-node-source/README.md)。

`GraphLowering` 继承 `torch.fx.Interpreter`。它按拓扑顺序遍历 FX 节点，用环境表保存
“FX node → 当前 Lowering 结果”，使后面的节点能取回前面节点留下的 IR 对象。

每个节点实际经过同一层外围处理：

```text
run() 取出下一个 FX node
        ↓
run_node(node) 建立本节点上下文
        ↓
按 node.op 分派
  ├─ placeholder：登记输入
  ├─ call_function：调用 target 对应的 lowering
  └─ output：整理图输出和输入写回
        ↓
run_node 补充来源、布局要求、读者和符号约束
        ↓
结果写入 env；普通节点继续下一项，output 则结束遍历
```

因此，placeholder、call_function 和 output 是 opcode 分支；`run_node` 包住每一次节点
处理，并不是排在 call_function 之后的第四个顺序阶段。

## 五份状态分别保存什么

| 状态 | key → value | 用途 |
| --- | --- | --- |
| Interpreter `env` | FX node → 本次解释结果 | 后续 node 按 FX 引用取得前面构造的对象 |
| `graph_inputs` | 输入名 → TensorBox 或符号表达式 | 记录生成程序的正式输入及其属性 |
| `graph_inputs_original` | Tensor 输入名 → 最初的 InputBuffer | 输入修改时找回调用者持有的 storage 身份 |
| `operations` / `buffers` | operation 顺序列表和名字表 | 记录已物化、需要 Scheduler 处理的工作 |
| `graph_outputs` | 顺序列表中的 IR 或常量结果 | 记录单图的普通返回值 |

`env` 连接 FX 节点；其余四份状态描述整张 Lowering 图的输入、工作和输出。一项延迟计算
可以先进入 env 而不增加 operation。

## 同一案例怎样走完整个解释器

独立实验 [`experiments/04_graph_lowering_nodes.py`](../../experiments/04_graph_lowering_nodes.py)
编译：

```python
def fn(x, y):
    a = x + y
    b = torch.relu(a)
    return b * 2
```

逐节点状态为：

| FX node | opcode 分支交出的结果 | buffers | operations |
| --- | --- | ---: | ---: |
| `arg0_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `arg1_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `aten.add.Tensor` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.relu.default` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.mul.Tensor` | `TensorBox → StorageBox → ComputedBuffer` | 1 | 1 |
| `output` | 返回 `None`；`graph_outputs` 指向 `ComputedBuffer` | 1 | 1 |

两个 placeholder 只登记输入。add 和 relu 留下可继续组合的位置规则。mul 是 output 的直接
输入，必须拥有稳定 storage，`run_node` 因而让最终 Pointwise 物化为 `ComputedBuffer
buf0`，并登记第一项 operation。output 把 buf0 加入 `graph_outputs`，本层出口是一项待
调度工作和一项图输出。

## placeholder：把样例属性变成输入接口

Tensor placeholder 读取样例的 device、dtype、size 和 stride，建立对应的输入描述：

```text
FakeTensor / tensor：shape=(3,), stride=(1,), dtype=float32, device=cpu
        ↓
FixedLayout(cpu, float32, size=[3], stride=[1])
        ↓
InputBuffer("arg0_1", layout)
        ↓
TensorBox(StorageBox(InputBuffer(...)))
```

这个 TensorBox 进入 env；InputBuffer 同时登记到 `graph_inputs` 和
`graph_inputs_original`。前者供生成程序读取输入，后者在 mutation 出口找回必须写回的
原 storage。

动态维度可用符号 `s0` 表示；Python 标量或符号标量也可作为符号表达式进入 env；`None`
则按输出合同保存为相应常量结果。各输入类型、符号约束和失败出口见
[输入登记实现](01-node-entry-source/README.md)。placeholder 只建立运行时读取说明，不读取
样例 tensor 的真实元素。

## call_function：把 FX target 分派给处理器

处理 add 时，FX 参数还是 placeholder 节点引用。解释器先从 env 解引用：

```text
FX：add(arg0_1_node, arg1_1_node)
        ↓
IR 参数：add(TensorBox(InputBuffer arg0_1),
             TensorBox(InputBuffer arg1_1))
```

随后根据 target 选择处理器。普通 ATen target 从 `lowerings` 注册表取得 lowering；
`operator.getitem` 直接从已有容器结果取项；pattern handler 按自身合同建立 IR；允许的
未知算子可留下 fallback operation。没有处理器又不允许 fallback 时，本节点从 Lowering
失败退出，并报告 target 与参数上下文。

贯穿案例的 add、relu、mul 依次在 env 中形成：

```text
add  → i ↦ load(x,i) + load(y,i)
relu → i ↦ max(add_rule(i), 0)
mul  → i ↦ relu_rule(i) * 2
```

每一步取回前一步的规则并接上新计算。完整分派顺序、fallback 和错误分支见
[算子与常量分派实现](02-dispatch-source/README.md)。

## run_node：为每个分支补上全图约束

`run_node` 在 opcode 处理器前后维护本节点与整张图的接口。它会把当前 FX node 的来源
附到新 IR 上，满足输出或消费者提出的布局要求，登记结果读取了哪些 buffer，并保存本
节点新产生的符号及运行时约束。

贯穿案例中，add 和 relu 没有独立输出身份，可以继续保持 Pointwise；mul 直接连接
output，必须成为可返回的 storage：

```text
TensorBox → StorageBox → Pointwise
                  │ realize
                  ▼
TensorBox → StorageBox → ComputedBuffer(name=buf0, data=Pointwise)
```

物化同时登记 buffer 和 operation，所以实验在 mul 后从 0 变为 1。其他触发条件、特殊
dtype、固定 stride、多用户复杂度、旧值读者和 unbacked symbol 断言都在
[`run_node` 实现](04-run-node-source/README.md)逐段展开；存储取舍由
[存储与数据版本](../03-storage-layout/README.md)解释。

## output：固定返回 storage 和输入写回

output 从 env 取得返回值，统一其容器形式，并让 Tensor 结果拥有可引用的 storage。贯穿
案例的 `graph_outputs` 最终保存 `ComputedBuffer buf0`；`output()` 自身返回 `None`，因为
它已经把结果写进图级状态，后面没有 FX 节点需要消费它。

若输入在图中被修改，output 还要登记一项把新值写回
`graph_inputs_original[name]` 所指原 storage 的 mutation operation。用户函数
`old=x+1; x.copy_(y); return old,x` 在单图合同中把 old 作为普通 graph output，把 x 的
变化作为输入副作用交回；外层编译合同再恢复用户可见的 x alias。完整的返回类型、stride
调整、mutation 和 finalize 过程见
[输出与输入修改实现](03-output-source/README.md)。

## 本层出口

贯穿案例离开解释器时：

```text
env：每个已处理 FX node 对应自己的 Lowering 结果
graph_inputs：arg0_1、arg1_1 的输入接口
operations/buffers：一项产生 buf0 的 Pointwise 工作
graph_outputs：buf0
```

`GraphLowering.codegen()` 随后把 `operations` 交给 Scheduler。若请求改成
`a=x+y; return a,relu(a)`，a 本身也有输出身份，会在 relu 之外拥有可返回 storage；这个
变化来自 users 和 output 合同，而不是 add 算式本身改变。
