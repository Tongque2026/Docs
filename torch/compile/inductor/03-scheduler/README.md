# Scheduler：把 IR 工作排成可生成的执行计划

[上一篇：Lowering](../02-lowering/README.md) · [返回总览](../README.md) · [节点与依赖](01-dependencies/README.md) · [排序与生存期](02-order-lifetime/README.md) · [融合](03-fusion/README.md) · [后端分派](04-backend-dispatch/README.md) · 下一篇：[Codegen](../04-codegen/README.md)

Lowering 已把 FX 节点变成 IR operation。Scheduler 接住这些 operation，先找出它们之间的读写关系，再删除不需要的工作、安排合法顺序、合并适合一起生成的工作，最后把结果依次交给相应后端。

这一步不计算 Tensor 的值，也不决定 `relu` 的数学含义。它处理的是“哪些工作必须先做、哪些可以合在一起、每份中间结果最后在哪里使用、每项工作交给谁生成”。

## 先认识 Scheduler 操作的对象

假设 Lowering 留下四项工作：

```text
op0：普通逐元素计算，读取 x，写 a
op1：普通逐元素计算，读取 a，写 b
op2：调用外部 mm，读取 b 和 weight，写 c
op3：普通逐元素计算，读取 c，写 output
```

Scheduler 为每项 operation 建立一条 scheduler node。记录中保留原 operation，还加入本阶段需要的信息：读哪些 buffer、写哪些 buffer、在哪个 device 上运行，以及它属于普通计算、外部调用还是模板计算。

| Lowering 留下的 operation | Scheduler 中的节点 | 后续生成方式 |
| --- | --- | --- |
| `ComputedBuffer` | `SchedulerNode` | 交给设备后端生成普通计算 |
| `TemplateBuffer` | `SchedulerNode`，保留 template 身份 | 交给 template 分支 |
| `ExternKernel` | `ExternKernelSchedulerNode` | 生成已有算子的调用点 |
| no-op operation | `NopKernelSchedulerNode` | 维持必要关系，不生成普通 kernel |

`Scheduler._init()` 建立这些节点以及按 operation 名、buffer 名查找节点的索引。此后 Scheduler 改变的是节点的关系、顺序和分组，不改 operation 所表示的数学计算。

## 一条计划怎样形成

四项工作会经历下面的状态变化：

```text
IR operations
      │
      ▼
建立 scheduler nodes 和读写边
      │  op0 → op1 → op2 → op3
      ▼
拓扑排序并删除无用节点
      │  每个消费者仍在生产者之后
      ▼
融合普通节点，整理最终节点序列
      │  [op0, op1] → op2 → op3
      ▼
在最终序列上记录每份数据的最后使用位置
      │  a 在融合组内用完，b 由 op2 最后读取，c 由 op3 最后读取
      ▼
按 device 和节点类型交给后端
```

这里要特别注意：排序和删除发生在融合前；最终的 `last_usage` 在融合以及后续节点整理完成后计算。这样记录的“最后使用者”对应的是最终交给生成阶段的节点，而不是融合前已经被合并掉的节点。

下面沿同一组 `op0` 到 `op3` 看每一步交出什么。

## 建立依赖：由 buffer 的生产和读取得到先后关系

`op0` 写 `a`，`op1` 读 `a`，所以 `op1` 必须在 `op0` 之后。相同规则继续得到：

```text
op0 --a--> op1 --b--> op2 --c--> op3
```

这些边既限制执行顺序，也给融合提供关系依据。例如 `op0` 和 `op1` 是直接相连的普通逐元素工作，可以继续检查是否适合融合；`op2` 是外部调用，节点类型已经表明它不能被当作普通逐元素循环处理。

写回同一份存储时，Scheduler 还要区分修改前后的逻辑版本。例如：

```text
m0：读取旧 x，写 old_result
m1：读取 y，把结果写回 x
m2：读取写回后的 x，写 output

m0 --保护旧值读取--> m1 --提供 x 的新版本--> m2
```

第一条边保证 `m0` 读完旧值以后才能写回，第二条边保证 `m2` 看到写回后的值。Scheduler 用 mutation 的名字映射和附加依赖保存这两个次序，避免把“同一存储”误当成“同一时刻的值”。具体怎样构造这些边，留到[节点与依赖](01-dependencies/README.md)。

## 排序和删除：先得到一条合法的活节点序列

依赖边建立后，拓扑排序保证每个生产者都排在消费者之前。若一项工作没有使用者、不是图输出，也没有必须保留的副作用，死节点删除会把它移走；若删除它以后，它的生产者也变得无用，删除会继续向前传播。

对当前案例，四项工作都通向 `output`，因此得到的活节点序列仍是：

```text
op0, op1, op2, op3
```

这一步还会计算祖先关系，供后面的候选判断使用：`op3` 依赖 `op2`，就不能为了凑一个融合组而越过 `op2`。排序规则、死节点删除和祖先信息在[排序与生存期](02-order-lifetime/README.md)展开。

## 融合：把可共同生成的工作变成一个节点

Scheduler 检查候选节点是否有可利用的数据关系、循环范围是否兼容、device 和后端是否一致，以及节点类型是否允许融合。

在这个案例中，假设 `op0` 与 `op1` 都是范围相同的普通逐元素计算，后端也允许融合，它们会变成一个融合节点：

```text
融合前：op0 → op1 → op2 → op3
融合后：[op0, op1] → op2 → op3
```

融合节点内部先由 `op0` 算出 `a[i]`，紧接着由 `op1` 使用它算出 `b[i]`。`a` 不必成为两个独立 kernel 之间的完整中间结果。`b` 仍要作为融合节点的输出，因为外部的 `op2` 需要读取它。

`op2` 保持独立：它是 extern node。`op3` 虽然也是普通计算，却依赖 `op2` 的结果，不能跨过这个边界并入前面的组。候选怎样产生、哪些检查会接受或拒绝融合，在[融合](03-fusion/README.md)展开。

TemplateBuffer 进入 Scheduler 后同样保留自己的身份。它可以按后端规则吸收合法的 epilogue，但不会被当作任意普通节点处理；最终仍沿 template 分支交给对应后端。本篇只追踪这个身份怎样影响分组与出口，模板 kernel 怎样生成属于 Codegen。

## 在最终节点上记录最后使用位置

融合和节点整理结束后，Scheduler 才调用 `compute_last_usage()`。对于当前的最终序列：

```text
[op0, op1] → extern op2 → op3
```

各中间结果的最后读取位置是：

| 数据 | 最后读取位置 |
| --- | --- |
| `a` | 融合节点内部成员 `op1`；这是 kernel 内部的最后使用 |
| `b` | extern `op2` |
| `c` | `op3` |

这些信息记录在调度节点上并交给 Codegen。Scheduler 到这里回答“最后在哪里用”，不在本篇展开 Codegen 怎样据此生成释放或复用存储的代码。

如果 `b` 同时也是图输出，它就不能只按 `op2` 的读取结束生存期。输出仍要引用这份数据，Scheduler 会把输出需求计入使用关系，并保留 `b` 的输出身份。

## 后端分派：按最终顺序交出节点

`Scheduler._codegen()` 按最终节点序列前进。普通节点交给当前 device 的普通后端；extern 节点生成已有调用；template 节点走 template 分支；device、extern 或 template 边界会让当前后端先结束正在积累的生成组。

当前案例的交接顺序是：

```text
普通融合节点 [op0, op1]
        │
        ├─ 先结束当前普通生成组
        ▼
extern op2
        ▼
新的普通生成组 op3
```

因此 Scheduler 的出口不是运行结果，而是一串有序的生成动作，以及附在最终节点上的依赖和最后使用信息。后端此时知道先生成包含 `op0/op1` 的普通计算，再生成 `op2` 的外部调用，最后生成 `op3`。

不同节点如何选择后端、何时结束当前组以及 device 切换如何处理，在[后端分派](04-backend-dispatch/README.md)展开。

## 用条件变化检查整条链

仍看 `[op0, op1] → extern op2 → op3`。

如果增加一项读取 `a` 的旁路工作，依赖图会多出 `op0 → 旁路节点`。Scheduler 必须重新排序，并重新判断旁路节点能否与原来的普通节点共同融合；最终节点改变以后，`a` 的最后使用位置也随之重算。

如果把 `op3` 放到另一 device，`op2 → op3` 的数据依赖不变，但两者不能组成同一 device 上的生成组。分派时会在边界结束当前组，再切换到另一设备的后端。

判断一个新案例时，可以沿实际状态变化追问：谁生产、谁读取；依赖允许哪些顺序；哪些活节点能组成一个生成单元；最终节点上每份数据最后在哪里使用；每个节点交给哪个后端。[节点与依赖](01-dependencies/README.md)、[融合](03-fusion/README.md)和[后端分派](04-backend-dispatch/README.md)分别展开相应工作；[排序与生存期](02-order-lifetime/README.md)同时解释融合前的排序和融合后的最后使用记录。
