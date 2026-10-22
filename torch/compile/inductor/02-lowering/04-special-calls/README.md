# 特殊调用：选择实现、保留外部调用、组织控制流子图

[返回 Lowering](../README.md) · [上一篇：存储与数据版本](../03-storage-layout/README.md) · [返回总览](../../README.md)

实现叶子：[候选、fallback 与控制流构造源码](01-special-source/README.md)。

Pointwise 和 Reduction 用逻辑索引描述通用循环；矩阵乘、没有原生 lowering 的 ATen
算子以及 `cond/while_loop` 需要保留更专门的执行身份。Lowering 为它们建立三类
operation：

```text
专用算子       → 候选/选择结果 → ExternKernel 或 TemplateBuffer
允许外部执行   → operator+参数 → FallbackKernel + MultiOutput
高阶控制流     → Lower 子图并核对接口 → Conditional / WhileLoop + MultiOutput
```

三条路径都要交出明确的 inputs、outputs、layout、device 和副作用，使 Scheduler 可以把它们
放进依赖图；它们的内部计算不必改写成普通 Loops。

## 专用算子：Lowering 交出候选或具体实现身份

矩阵乘已有库实现和专用模板。Lowering 先确定输入、输出 layout 和问题规模，再建立当前
设备与配置允许的候选。候选选定后有两种主要出口：

| Lowering 出口 | 图中的含义 | Scheduler 收到什么 |
| --- | --- | --- |
| `ExternKernelOut` / `ExternKernelAlloc` | 调用已有函数或库实现 | 一项有输入和输出的 extern operation |
| `TemplateBuffer` | 一份带输入、输出 layout 和生成配置的模板工作 | 一项 template operation |
| `MultiTemplateBuffer` | 多个实现身份和选择信息仍被保留 | 一项稍后还需完成选择的 template operation |

独立实验 [`experiments/07_special_lowering_paths.py`](../../experiments/07_special_lowering_paths.py)
在当前 CPU 配置对 4×4 float32 mm 观察到：

```text
TensorBox → StorageBox → ExternKernelOut(name=buf0)
operations = [ExternKernelOut]
```

本例的 Scheduler 因而收到一个 extern node。其他设备和配置可产生 TemplateBuffer 或
MultiTemplateBuffer；Lowering 到此只交出候选/operation 身份。后续怎样结合 epilogue
完成选择属于 Scheduler，怎样渲染模板和发出调用属于 Codegen。

## fallback：保留 operator 和返回接口

没有原生 Inductor 计算实现、但允许在已编译图中调用的算子，可以建立
`FallbackKernel`。Lowering 用 fake 输入取得输出的 shape、stride、dtype 和容器结构，
保存 operator 与参数，再为每个 Tensor 返回项建立 `MultiOutput`。

例如：

```python
x = [0.5, 1.5, 1.8, 3.2]
torch.histc(x, bins=4, min=0, max=4)
```

产生：

```text
FallbackKernel(name=buf0, op=aten.histc.default)
  └─ MultiOutput(name=buf1, 对应 Tensor 输出)
```

运行结果为 `[1,2,0,1]`。buf0 表示一次共同调用，buf1 是后继节点引用的 Tensor 结果；
它们不会让同一个 operator 执行两次。外部调用的输入在 Lowering 中先成为可传入的真实
数据，Scheduler 再把这项 extern operation 排到生产者之后。

operator schema 还说明返回值是否 alias 输入、调用是否修改输入。Lowering 只有在能明确
表达这些关系时才交出 fallback operation；不支持的接口在本层失败。完整参数拆装、schema
检查和多输出构造见实现叶子。

## cond：两支子图形成同一个父图接口

```python
def true_fn(v):  return v + 1
def false_fn(v): return v - 1
return torch.cond(pred, true_fn, false_fn, [x])
```

输入 `x=[2,4]` 时，predicate 为真得到 `[3,5]`，为假得到 `[1,3]`。Lowering 分别处理
true/false 子 GraphModule，得到两份子图输出，再核对输出数量以及每项的 size、stride、
device、dtype 和 offset。接口一致时，父图登记：

```text
true_graph  → ComputedBuffer(add)
false_graph → ComputedBuffer(sub)
parent      → Conditional(buf0) → MultiOutput(buf1)
```

`Conditional` 是一次父级控制流 operation；`MultiOutput` 让父图后继引用统一结果。Lowering
编译两支接口，运行时选择哪一支由后续生成的控制流决定。两支 shape 或 dtype 等接口不
一致时，本层无法建立统一 MultiOutput，构造在进入 Scheduler 前失败。

## while_loop：body 输出成为下一轮输入

`while_loop` 接收 condition 子图、body 子图、循环携带值和只读附加输入。Lowering 分别
处理 condition/body，并检查 condition 交出一个 bool 标量，body 的输出数量及每项
size、stride、device、dtype 与 carried inputs 对应。

设 carried 初值为标量 tensor `v=0`，condition 是 `v<3`，body 返回 `v+1`：

```text
进入循环：carried slot 0 = tensor(0)
第 1 轮：body 输出 tensor(1) → 下一轮 slot 0
第 2 轮：body 输出 tensor(2) → 下一轮 slot 0
第 3 轮：body 输出 tensor(3) → condition 为假，成为父图结果
```

父图登记一项 `WhileLoop` operation，并用 `MultiOutput` 表示最终 carried value。若 body
把 `[2,3]` tensor 变成 `[3,2]` 后返回，这份结果无法作为原 carried slot 的下一轮输入，
本层在接口检查时拒绝。

## 三条路径的 Lowering 出口

| 路径 | 本层新增状态 | 交给 Scheduler |
| --- | --- | --- |
| 实现选择 | 候选输入/layout，或具体 extern/template operation | ExternKernel 或 Template scheduler node |
| fallback | operator、参数结构、Tensor outputs、alias/mutation 合同 | extern node 与 MultiOutput 引用 |
| cond/while | 子 GraphLowering、父级输入、统一输出槽位 | Conditional/WhileLoop extern node 与 MultiOutput |

Scheduler 读取这些接口来建立依赖和分派类型，不重新推导 histc 数学、模板源码或控制流
分支内容。具体构造步骤见[实现叶子](01-special-source/README.md)。
