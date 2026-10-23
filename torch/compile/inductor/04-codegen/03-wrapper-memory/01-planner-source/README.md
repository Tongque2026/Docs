# Wrapper 复用规划源码：先记录生存期，再决定同一块 storage 给谁

[返回 Wrapper 与内存计划](../README.md) · [上一叶子：CPU/Triton backend](../../02-backends/01-backend-source/README.md)

backend kernel 会读写有名字的 tensor，但 wrapper 还要决定每个名字背后使用哪块
storage。下面这个请求中，A 由 `kernel0` 写入，`kernel1` 最后一次读取 A；随后
`kernel2` 才需要同规格的 C：

```text
申请 A → kernel0 写 A → kernel1 读 A → 释放 A → 申请 C → kernel2 写 C
```

若 A、C 的 device、dtype 和 storage 容量兼容，C 可以接管 A 的 storage，最终
wrapper 不必为 C 再申请一块空间。但这个决定不能在 A 刚创建时作出：当时还不知道
A 的最后一个使用者，也不知道后面是否真的出现 C。

源码先把这些动作保存成可修改的、尚未输出 Python 的 wrapper planning line 对象。
`Buffer` 记录一个逻辑结果的名字、
device、dtype、size 和 layout；`AllocateLine(A)`、`FreeIfNotReusedLine(A)`、
`ReuseLine(A,C)` 分别表示申请、候选释放和接管。`ReuseLine` 有两种来源：codegen
事先确定的 inplace 共享合同，以及默认 planner 把一条 free 与后来的 allocate 配对后
产生的替代行。这些都还是生成期对象，不是正在执行的 Python tensor。planner 修改
整条行列表，最后每一行的 `codegen` 才写出
`empty_strided`、`del` 或下面的复用代码：

```python
C = A
del A  # C 与 A 指向同一块 storage，不复制元素
```

本篇有两条互斥的规划路径，共用同一份 wrapper 行输入：

```text
                         ┌─ 推理且 config.memory_planning=True
全部 wrapper lines ─────┤    pool planner：合并 inplace 名字，计算 live range，
                         │    决定 pool 与 byte offset，再生成 pool view
                         │
                         └─ 其他情况（训练或未开启 memory_planning）
                              默认 planner：按 reuse key 用 LIFO 栈配对 free/allocate
```

它们不是先执行默认复用、再执行 pool 的两个阶段。wrapper 的生成入口只选择其中
一条：推理且开启 `memory_planning` 时调用 `MemoryPlanner.plan`；否则调用
`memory_plan_reuse`。两条路径最终都交回一份可按顺序 codegen 的 wrapper lines。

后文还会反复使用一条固定的 8×8 float32 案例：

```python
first = mm(x, weight0)          # buf0，64 个 float32
activated = relu(first)         # buf1，与 buf0 显式 inplace 共享
second = mm(activated, weight1) # buf2；此时仍要读取 buf1
result = sin(second)            # buf3，与 buf2 显式 inplace 共享；buf3 是输出
```

进入规划器的关键时间线一次性列全如下。普通调用行把 planning 段隔开；显式
`Reuse` 是 codegen 在规划前已经确定的共享合同，不是默认 LIFO 临时配对：

```text
Allocate(buf0)
call mm(x,weight0,out=buf0)
Reuse(buf0,buf1)
call fused_relu(buf1)
Allocate(buf2)
call mm(buf1,weight1,out=buf2)   # buf1 的最后读取
Free(buf1)
Reuse(buf2,buf3)
call fused_sin(buf3)
return buf3
```

默认路径保留两条显式 reuse，并为 buf0、buf2 各做一次真实 allocation。pool 路径
把 `{buf0,buf1}` 合成中间组 G1，把 `{buf2,buf3}` 合成输出组 G2，再以组为单位
计算生存期、pool 和销毁位置。K01 先用更小的 A/C 案例解释默认 LIFO；K03、K04、
K06 都只是在不同实现层面继续代入这条固定时间线。

## K01：把分配和释放先保存为可修改的时间线

假设 wrapper 按顺序需要三个 float32 CPU buffer，三者容量相同：

```text
1 Allocate(A)
2 call kernel0(..., A)
3 Free(A)
4 call unrelated_kernel()
5 Allocate(C)
```

记录第 1 行时不能把 A 的 storage 直接承诺给 C，因为 C 尚未出现；记录第 3 行时也
不能立即输出 `del A`，否则第 5 行看不到可接管的 storage。Inductor 因而先把分配和
释放保存成携带 IR buffer 的对象，等 Scheduler 把全部 wrapper 行写完后再扫描。

默认 planner 的状态是一张按兼容 key 分组的栈：

```text
reuse_pool[(device, dtype, symbolic_storage_size)] -> [尚未被接管的 free 行]
```

上例的状态变化是：

| 扫描到 | key 栈 | 原时间线怎样改 |
| --- | --- | --- |
| `Allocate(A)` | 空 | 没有旧块，暂时保留真实 allocation |
| `Free(A)` | `[Free(A)]` | free 行先不删除，只进入候选栈 |
| `Allocate(C)` | 弹出 A，栈恢复空 | 标记 `Free(A).is_reused=True`；把本行换成 `Reuse(A,C)` |

第二遍生成代码时，已复用的 `Free(A)` 不输出 `del A`；`Reuse(A,C)` 输出新名字接管
旧 storage，并按输出身份决定是否删除旧名字。若没有 C，free 行保持未复用，第二遍
才输出 `del A`。

对上例，扫描开始时 A、C 都是 `Buffer`；行列表依次含 `AllocateLine(A)`、调用
文本、`FreeIfNotReusedLine(A)`、调用文本、`AllocateLine(C)`。planner 修改这些
行对象，最后生成阶段才写 Python 代码。

栈采用 LIFO。若同一 key 依次释放 A、B，再分配 C，C 接管最近释放的 B；A 仍留在
候选栈，可供后一个 D 使用。这不是按名字匹配，而是同 device、dtype、简化后 storage
容量表达式的候选中优先取最近一项。

普通文本调用行不会清空候选栈，它只是保持执行顺序。进入子图时则新建独立
`MemoryPlanningState`，退出时收起该状态，避免子图内部 free 被外层 allocation 直接
接管。扫描末尾还会去掉不属于图输出、且连续落在末尾的无效 planning line；这些行后面
已经没有消费者或新 allocation，保留它们不能产生可观察结果。

### 条件对照

- C 的 dtype 改成 float16：reuse key 不同，A 留在原 key 栈，C 保留真实 allocation。
- C 形状改变但 storage 容量、device、dtype 相同：key 仍可能匹配；生成阶段用
  reinterpret view 表达 C 的 shape/stride，是否可复用还要通过身份过滤。
- A 是图输入：wrapper 可以在最后使用后删除局部名，但 `can_reuse` 禁止把调用者的
  storage 放进普通中间复用池。

### K01 迁移题

时间线改为 `Free(A) → Free(B) → Allocate(C) → Allocate(D)`，四者 key 相同。
第一次 allocate 弹出 B，所以 C 接管 B；第二次再弹出 A，所以 D 接管 A。若 B 是
alias-output buffer，它的 free 不会入栈，则 C 接管 A，D 必须真实分配。这一变化来自
候选集合改变，而不是 C/D 名字或数值内容改变。

K01 建立了规划对象存在的原因和扫描状态。K02 将打开 allocation/free 入口与身份
过滤，解释哪些 line 根本不会进入上述候选池，以及 exact alias 与 reinterpret 两个出口。

## K02：先判断 buffer 是否拥有可接管的 storage

`codegen_allocation(buffer)` 并不等价于“写一行 empty”。它先判断这份 IR buffer 的
存储由谁拥有：

| 条件 | 采取的动作 | 原因 |
| --- | --- | --- |
| 名字已 removed 或已经登记 allocation | 直接返回 | 删除项和重复请求都不应再产生存储 |
| defining op 是 `ExternKernelAlloc` / `MultiOutput` | 只登记名字，不写 `AllocateLine` | 存储由外部调用或多输出操作建立 |
| `MutationLayoutSHOULDREMOVE` / `NoneLayout` | 返回 | 前者应由别名/修改关系处理，后者本就没有普通 tensor storage |
| `NonOwningLayout` | 递归登记 backing buffer，再写 deferred alias | 当前名字只是已有 storage 上的 view |
| 普通 owning layout | 写入 `AllocateLine(self, buffer)` | 到全局规划时才决定真实分配或复用 |

例如 C 是 A storage 上 shape=(2,3)、stride=(1,2) 的非 owning view。入口先确保 A 的
owning buffer 已进入时间线，随后为 C 写 alias 引用；它不会为 C 再建一个普通
`AllocateLine`。这和“C 后来复用一个同容量旧 buffer”不同：前者的共享关系在 IR
layout 中已经确定，后者要等生存期扫描才能决定。

### free 入口把“删除局部名”和“storage 可复用”分开

`codegen_free(buffer)` 有三种出口：

1. `InputBuffer` 立即写普通 `del name`。这只删除 wrapper 的局部 Python 引用；调用者
   仍持有输入 tensor，因此这块 storage 不进入 reuse pool。
2. `can_reuse(buffer)` 为 False 时不写 planning free。removed、graph input、常量、
   torchbind 常量、`never_reuse_buffers` 和已经释放过的名字都在这里被排除。
3. 普通中间结果加入 `freed`，写 `FreeIfNotReusedLine`，留到规划阶段决定是 `del`
   还是被后续 allocation 接管。

图输出的保护还依赖时间线合同，而不只靠 `can_reuse`：Scheduler 不应在返回前把仍是
输出的 storage 当普通死值释放；`memory_plan_reuse` 也不会把末尾属于 output name 的
planning line当作无效尾行删除。若一个旧名字本身是输出，`make_buffer_reuse` 在生成
新名字时不会输出 `del old`，使返回合同仍能观察旧名。

### 显式 inplace reuse 在扫描前就知道旧、新 storage 必须相同

某些 pointwise 已由前面的别名和生存期分析确定可以原地覆盖，入口会调用
`codegen_inplace_reuse(old,new)`。它先断言两者 reuse key 相同，然后：

```text
确保 old 已登记 allocation
old 加入 freed
new 加入 allocated
reuses[new] = old
写入 ReuseLine(old,new)
```

这条 `ReuseLine` 不是默认 planner 从“一个 free + 后一个 allocate”偶然配对出来的，
而是 codegen 已经给出的显式共享合同。后面的 pool planner 也据此把 old/new 合并为
同一个 `BufferGroup`。

### exact alias 与 reinterpret view

规划最终调用 `make_buffer_reuse(old,new,delete_old)`。dtype 必须相同；reuse key 在更
早阶段已经保证 device 与 storage capacity 兼容。生成端再比较 shape/stride：

- 完全相同：`new = old; del old  # reuse`；没有元素复制。
- shape 或 stride 不同：生成 `reinterpret_tensor(old,new_size,new_stride,0)`，让 new
  在同一 storage 上采用新逻辑视图，然后按条件删除 old 名。

假定 A 是 contiguous 2×3，元素 stride=(3,1)，storage capacity 为 6；C 是 3×2，
stride=(1,3)，capacity 仍为 6。同 device/dtype 下 key 可以相同，但 C 不能直接把 A
当成相同布局使用。reinterpret 后的具体位置是：

| C 逻辑位置 | storage 元素位置 |
| --- | --- |
| `[0,0]` | 0 |
| `[0,1]` | 3 |
| `[1,0]` | 1 |
| `[2,1]` | 5 |

复用的是六元素 storage，改变的是 C 的位置解释。若 C 需要七元素容量，reuse key 已在
扫描阶段不同，不会走到 reinterpret 分支。

### alias output 与 MultiOutput 的 free 分支

即使一条 `FreeIfNotReusedLine` 已经存在，`plan()` 还要检查 buffer 是否有
`inputs_that_alias_output`，或 layout 是否是 `MultiOutputLayout`。命中任一项时，它
保留 free line 但不 push 入候选栈；第二遍因此只执行正常 free 语义，不把与输出共享的
storage 交给无关新 buffer。removed buffer 则直接把 line 换成 `NullLine`。

### K02 状态推演

给定时间线：输入 X 最后使用、普通 A 最后使用、alias-output B 最后使用，随后申请
同 key 的 C、D：

```text
Free(X) -> 立即文本 del X，不进入 planning state
Free(A) -> push A
Free(B) -> 保留 line，但 alias-output 条件使 B 不 push
Allocate(C) -> pop A，得到 Reuse(A,C)
Allocate(D) -> 栈为空，D 保留真实 allocation
```

如果 C 与 A 同容量但 stride 不同，最后一行从 exact alias 改为 reinterpret view；如果
C dtype 改变，连 A 都不会被弹出。这里分别由身份、reuse key、布局三个层次作决定，
不能用“元素数相同”替代。

K02 到此完成。K03 将逐句保留默认 planner 声明范围的完整源码，并把开头固定案例的
`buf0/buf1`、`buf2/buf3` 代入 `MemoryPlanningState` 与两遍 codegen。


## K03：默认 planner 的完整源码与两次 mm 状态

### reuse key 与 `MemoryPlanningState`

下面五个短方法属于同一个 `MemoryPlanningState` 类。新状态只有两个字段：
`reuse_pool` 是空的 `key → free-line 栈`，`total_allocated_buffer_size=0`。planning
line 的字段初态也先固定：`FreeIfNotReusedLine(node).is_reused=False`；
`ReuseLine(old,new).delete_old=True`；`AllocateLine` 只持有待申请的 `node`。
第一次扫描调用各行的 `plan(state)` 修改这些字段或返回替代行，第二次才调用
`codegen`。因此同名的 `plan/codegen` 是不同 line 子类实现同一两遍合同，不是一类
里反复覆盖的方法。

```python
# 教学源码：以下方法保持产品源码原顺序；本行仅标识教学副本。
def buffer_reuse_key(node: ir.Buffer) -> ReuseKey:
    return (
        node.get_device(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(node.layout.storage_size())),
    )

def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[
            ReuseKey, List[FreeIfNotReusedLine]
        ] = collections.defaultdict(list)
        self.total_allocated_buffer_size: int = 0

def __contains__(self, key: ReuseKey) -> bool:
        return bool(self.reuse_pool.get(key, None))

def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        assert not item.is_reused
        self.reuse_pool[key].append(item)
```

### 三类 planning line

下面其实连续摘录了四个类的方法，而不是同一个类把 `plan/codegen` 定义四次。按出现顺序分别是：共同基类 `MemoryPlanningLine`、申请新空间的 `AllocateLine`、把空间放回候选池的 `FreeIfNotReusedLine`、让新 buffer 接管旧空间的 `ReuseLine`。代码内的分隔注释标出每组方法所属的类。

```python
# 教学源码：以下方法保持产品源码原顺序；本行仅标识教学副本。
# MemoryPlanningLine：所有计划行的默认行为
def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

def codegen(self, code: IndentedBuffer) -> None:
        """Second pass to output code"""

def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        args: List[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"

# AllocateLine：为 self.node 找可复用空间，否则保留一次新分配
def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        if self.node.get_device().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)

# FreeIfNotReusedLine：最后一次使用后，把 self.node 放入候选池
def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))

# ReuseLine：输出“旧空间改作 reused_as”的代码
def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )
```

### 两遍扫描与子图状态

先固定一个独立的最小默认复用例；它不是开头两次 mm 案例中的同名 buffer。
`self.lines` 依次是 `Allocate(buf0[8×8]) → kernel0 → Free(buf0) →
Allocate(buf1[8×8]) → kernel1 → Free(buf1)`，输出名集合不含 buf0/buf1，
`allow_buffer_reuse=True`，初始候选池为空，也没有嵌套子图。第一遍执行
`line.plan(...)`：buf0 首次分配，Free(buf0) 把它压入候选池，Allocate(buf1)
取得同 key 的 buf0 并改成 ReuseLine。第二遍才调用每行 `codegen()`，输出一次
alloc、一次 reuse 和两次 kernel 调用；已经被复用的 buf0 不再生成真正的 free。

```python
# 教学源码：以下方法保持产品源码原顺序；本行仅标识教学副本。
def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        assert len(planning_states) == 0

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

        self.allow_stack_allocation = (
            self.allow_stack_allocation is not False
            and config.allow_stack_allocation
            and total_allocated_buffer_size <= MAX_STACK_ALLOCATION_SIZE
        )
```

### allocation/free/inplace 入口与复用代码生成

```python
# 教学源码：以下方法保持产品源码原顺序；本行仅标识教学副本。
def make_buffer_reuse(self, old: ir.Buffer, new: ir.Buffer, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            if old_name in self.stack_allocated_buffers:
                self.stack_allocated_buffers[new_name] = new
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call
        )
        if reinterpret_view in self.stack_allocated_buffers:
            self.stack_allocated_buffers[new_name] = new
        return f"{self.declare_maybe_reference}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

def codegen_allocation(self, buffer: ir.Buffer):
        name = buffer.get_name()

        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)
        if isinstance(
            buffer.get_defining_op(),
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            return
        if isinstance(layout, ir.NoneLayout):
            return
        if isinstance(layout, ir.NonOwningLayout):
            assert isinstance(
                layout.view, ir.ReinterpretView
            ), f"unexpected {type(layout.view)}: {layout.view}"
            assert isinstance(layout.view.data, ir.StorageBox), type(layout.view.data)
            assert isinstance(layout.view.data.data, ir.Buffer), type(layout.view.data)
            self.codegen_allocation(layout.view.data.data)
            self.codegen_deferred_allocation(name, layout)
            return

        self.writeline(AllocateLine(self, buffer))

def codegen_free(self, buffer):
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        self.writeline(FreeIfNotReusedLine(self, buffer))

def can_reuse(self, input_buffer, output_buffer=None):
        name = input_buffer.get_name()
        return not (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in V.graph.torchbind_constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        )

def codegen_inplace_reuse(self, input_buffer: ir.Buffer, output_buffer: ir.Buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))
```

### 20 个声明怎样共同完成两遍规划

`buffer_reuse_key` 使用化简后的符号 storage size 字符串，而非 size hint。若两个动态
尺寸在本次样例中碰巧都是 64，符号表达式分别是 `s0` 和 `s1`，key 仍不同；planner
不会因为样例值相等就把不同运行时容量混用。

第一次扫描调用每条 planning line 的 `plan`：

- `AllocateLine` 先处理 removed，再尝试从 state 弹栈；没有候选时保留自身，并对静态
  CPU shape 累加元素数，供后面的栈分配阈值判断。
- `FreeIfNotReusedLine` 对 alias-output/MultiOutput 直接保留但不入栈；普通项按 key
  push；removed 变 `NullLine`。
- 已明确的 `ReuseLine` 只检查 old/new removed 状态，保持原共享决定。

扫描遇到子图入口时压入新的 state，出口时弹出。所有 state 的
`total_allocated_buffer_size` 最后求和；只有配置允许、此前没有显式禁止、总静态 CPU
元素数不超过 `MAX_STACK_ALLOCATION_SIZE`，wrapper 才允许栈分配。这里累计的是保守
上界，嵌套 scope 不互相借 reuse pool，却都计入总量。

第二遍发生在 wrapper 输出各 line 时：保留下来的 allocation 调用
`make_buffer_allocation`；未复用 free 输出 `del`；被标记复用的 free 静默；ReuseLine
调用 `make_buffer_reuse` 选择 exact alias 或 reinterpret。`NullLine` 不输出内容。

### 固定案例的两组显式 reuse

开头两次 mm 案例的 pointwise 覆盖已经在 codegen 阶段被判为显式 inplace，因此进入
规划器的是两条 `ReuseLine`，而不是依靠 LIFO state 临时匹配：

```text
Allocate(buf0) -> mm 写 buf0
Reuse(buf0,buf1) -> relu 原地写同一 storage
Allocate(buf2) -> 第二次 mm 写新 storage
Reuse(buf2,buf3) -> sin 原地写同一 storage
```

第一次扫描时两条 ReuseLine 均原样保留。第二遍分别生成：

```python
buf1 = buf0; del buf0  # reuse
buf3 = buf2; del buf2  # reuse
```

第二次 mm 仍必须 `Allocate(buf2)`，因为执行该调用时 buf1 是输入；源码中的复用扫描
只能利用已经出现的 free 或显式共享合同，不能从未来借一块仍在使用的 storage。
本案例的生成产物包含这两条显式 inplace 路径；K01 的普通 free→allocate LIFO
配对则由当前源码和状态推演支持，二者不是同一个决定。

### 已删除的 buffer 会在哪几个步骤从规划时间线消失

入口可能先因为 `V.graph.removed_buffers` 不写 allocation；已经写入时间线后 buffer
又被优化删除，line 的 `plan` 会把它换成 `NullLine`。此外，`memory_plan_reuse` 在扫描
前从末尾连续弹掉非输出 planning line，因为末尾之后已没有调用或新 allocation 能
消费它们。输出 name 会阻止这项尾删，保留返回所需对象。三者分别发生在记录时、规划
时、扫描开始前，解决的不是同一个状态。

### K03 迁移推演

若在第一条 relu 后额外返回 `first`，buf0 的旧值成为图输出。原地覆盖会破坏该输出，
因此上游不应再交出 `Reuse(buf0,buf1)`；需要为 activated 使用独立 storage，或采用能
同时保留旧结果的其他安排。默认 planner 也不能把输出末尾 line 当无效项删除。

若只是把 buf1 的 layout 改成同容量转置，而旧值不再使用，显式 reuse key 仍可相同，
`make_buffer_reuse` 从 exact 分支改为 reinterpret；storage 身份保持，逻辑位置关系改变。
若 dtype 改变，`codegen_inplace_reuse` 的 key 断言先失败，不能等到生成字符串时补 cast。

K03 到此完成。K04 转入 pool planner，先建立 `BufferGroup` 与离散时间：为什么连续的
planning line 属于同一 timestep，普通 kernel 调用又怎样把两个使用阶段分开。

## K04：把多个逻辑名字合成一块 storage 的生存期

pool planner 不再逐次寻找“最近 free 的同规格 buffer”，而是先回答两个更全局的问题：

1. 哪些逻辑名字本来就指向同一块 storage？
2. 每块 storage 从 wrapper 的哪个阶段活到哪个阶段？

`config.memory_pool` 只在已经选择 pool planner 后决定各类 group 怎样组合；它不控制
前面“默认还是 pool”那次选择：

| 值 | 中间组 | 输出组 | 输出能否与中间组同池 |
| --- | --- | --- | --- |
| `none` | 可在已有空间中做时间复用，但 pool 不能向末尾扩容；放不下就新建 pool | 每个输出建立独立 pool | 否 |
| `intermediates` | 最后一个中间 pool 可扩容 | 每个输出建立独立 pool | 否 |
| `outputs` | 中间 pool 可扩容 | 多个输出按顺序追加到同一 output pool | 否 |
| `combined` | 中间与输出按同一排序和放置算法处理 | 不再单独分流 | 是 |

所有 pool 首先按 device 分组；dtype 不属于 `AllocationPools` 的分池 key。每个 tensor
view 仍保留自己的 dtype，容量统一换成 byte 数参与 offset 和 pool 大小计算。

`compute_buffer_groups` 顺序读取原 planning line。每个 `AllocateLine(A)` 建一个
`BufferGroup(A)`；遇到 `ReuseLine(A,B)`，就在 A 所在组的 `names` 追加 B，并让名字 B
也映射回同一个 group。最后只按对象身份去重 group，并检查其中任一名字是否为图输出。

固定案例因而得到两组：

```text
G1: node=buf0, names=[buf0, buf1], is_output=False
G2: node=buf2, names=[buf2, buf3], is_output=True
```

组仍以最初 owning allocation 的 node 表示容量和布局；后续 inplace 名只说明共享
storage。`ReuseLine` 不会新建 G3/G4，否则 planner 会把同一块物理内存错误计算两次。

### 从普通 planning line 转成 pool line

`convert_to_pool_lines` 使用上述 name→group 映射修改原时间线：

- `AllocateLine` → `AllocFromPoolLine(group)`；
- 可复用的 `FreeIfNotReusedLine` → `DeallocFromPoolLine(group)`；
- `ReuseLine` 仍保留，但将 `delete_old=False`，因为 pool 的整体销毁负责删除同组名字。

removed line 已在此前变成 `NullLine`，没有 group。外部调用自行创建的 storage 原本也
没有普通 AllocateLine，因此当前 TODO 明确不把这类 old name 强行纳入 reuse group。

### timestep 不是每一行加一，而是每一段 planning line 加一

`compute_live_ranges` 用 deque 扫描完整 wrapper 时间线。若队首是
`MemoryPlanningLine`，timestep 加一，并把后面连续的所有 planning line 都归入这个
时刻；其中只有 `PoolMemoryPlanningLine` 才调用 `group.update_usage(timestep)` 并保存
自己的 timestep。若队首是普通 kernel、guard 或文本行，只弹掉这一行；它不直接加
数字，但会切断前后两段 planning line，使下一段得到新 timestep。

因此时间数字表示“调用之间的内存状态边界”，而不是源文件行号。例如：

```text
planning: Alloc(G1)       -> t=1，G1 使用 1
ordinary: first mm
planning: Reuse(buf0,buf1)-> t=2，但不是 pool line，不单独更新 G1
ordinary: relu
planning: Alloc(G2)       -> t=3，G2 使用 3
ordinary: second mm
planning: Dealloc(G1), Reuse(buf2,buf3)
                           -> t=4，G1 使用 4；Reuse 不更新 G2
ordinary: sin / return
规划结束的输出延长点       -> t=5，G2 因 is_output 使用 5
```

G1 的 `update_usage` 先把初始 `(inf,-inf)` 变为 `(1,1)`，再扩成 `(1,4)`；G2 从
`(3,3)` 扩成 `(3,5)`。`LiveRange` 约定 begin inclusive、end exclusive：数字代表
两段普通执行之间的边界，t=4 的 dealloc 把结束边界记为 4，不表示 storage 还占用
边界之后的第 4 段。这里的离散端点主要用于比较：G1 与 G2 在第二次 mm 阶段重叠，
因此不能占 pool 的同一 offset；
G1 结束后出现的另一个中间 storage 才可能在时间上复用它的位置。

### `LiveRanges` 为什么允许多个不连续区间

单个 `LiveRange(begin,end)` 能做包含与并集；`LiveRanges` 接收多个区间，先按 begin
排序，再把相交或首尾接触的区间合并。`overlaps` 用两个 deque 比较当前最早区间：
若左区间结束仍晚于右区间开始，立即重叠；否则丢掉较早结束者继续。

具体代入：

```text
A = [1,4) 与 [7,9)
B = [4,7)
```

A 和 B 不重叠：边界 4、7 处前一段已经结束。若 B 改为 `[3,7)`，它和 A 的第一段
在 `[3,4)` 重叠。若一组区间输入为 `[1,3)、[2,5)、[8,9)`，构造后合并为
`[1,5)、[8,9)`。

### 输出为什么必须延长到规划末尾

输出 storage 在最后一次 kernel 写完后仍要交给调用者，不能以最后一个内部使用点为
结束。扫描完所有 line 后，planner 再令 timestep 加一，并对 `is_output=True` 的组执行
一次 `update_usage`。实验中的 G2 因而活到 t=5；若没有这一步，pool 可能在 return 前
把 buf3 所在区域交给另一个临时值。

### K04 迁移题

若函数同时返回 `first=buf0` 和最终 `buf3`，G1 因包含输出名而变成 output，结束点也
延长到 t=5；同时上游原本的 `Reuse(buf0,buf1)` 必须取消，否则 relu 会覆盖 first。
结果不是简单地给既有 G1 加一个标志，而是 group names 和原时间线都会变化。

若在 G1 释放之后、返回之前增加中间 G3，且 G3 的范围为 `[4,5)`，它与 G1 的
`[1,4)` 不重叠，可以在 K05 的 `TemporalSplit` 中共享 offset；它仍与 G2 的 `[3,5)`
重叠，不能占 G2 的位置。这个判断只依赖半开区间，而不是 buffer 名字或出现距离。

K04 到此完成。K05 将打开时间/空间分配树的完整实现，具体推演三个不同大小与生存期
的 allocation 怎样形成 `TemporalSplit`、`SpatialSplit` 和最终 byte offset。


## K05：时间复用与空间分割组成一棵 allocation tree

### 完整源码：live range、allocation 节点、时间/空间分割与 pool

先固定两种树节点的含义：

```text
TemporalSplit（时间分支）
├─ child A：只在一段时间存活
└─ child B：与 A 时间不重叠
结果：A/B 使用同一 offset；节点容量取各 child 最大值

SpatialSplit（空间分支）
├─ left：一段 byte 区间
└─ right：紧随其后的另一段 byte 区间
结果：两边可以同时存活；节点容量为对齐后的 left 加 right
```

进入源码再固定三块 allocation：A 大小 64 byte、live `[1,4)`；B 大小 32 byte、
live `[4,6)`；C 大小 48 byte、live `[2,5)`。三者 device 相同，pool 初始为空，
A 作为第一个 allocation 允许扩展 pool。A 与 B 时间不重叠，可以进入同一个
`TemporalSplit`、共享 offset 0，该 slot 容量取 max(64,32)=64；C 同时与 A、B
重叠，必须进入 `SpatialSplit` 的另一侧，offset 为 64；最终 pool 容量 112。
阅读各 `_allocate` 分支时始终用这三组 size/live range 判断 overlap、扩容与 offset。

```python
# 教学源码：LiveRange 完整产品定义；本注释仅标识教学副本。
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """

    begin: float  # int | +/-inf
    end: float  # int | +/-inf

    def contains(self, other: LiveRange):
        """Is other entirely within self"""
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        return self.end - self.begin

# 教学源码：LiveRanges 完整产品定义；本注释仅标识教学副本。
class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            assert self.ranges[-1].begin <= r.begin
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            assert left[0].begin <= right[0].begin
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        return self.ranges[0].begin

    @property
    def end(self):
        return self.ranges[-1].end

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"

# 教学源码：AllocationTreeNode 完整产品定义；本注释仅标识教学副本。
class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this bool.  Return True if
        an assignment was made.
        """
        return False

    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
        return self

    def is_empty(self):
        return False

# 教学源码：Allocation 完整产品定义；本注释仅标识教学副本。
class Allocation(AllocationTreeNode):
    """
    Represents memory allocated to a given node in the allocation pool.
    """

    node: ir.Buffer
    live_range: LiveRange
    size_hint: int
    symbolic_size: sympy.Expr
    allocated: bool = False
    pool: Optional[AllocationPool] = None
    offset: Optional[sympy.Expr] = None

    @property
    def device(self):
        return self.node.get_device()

    def get_live_ranges(self):
        return LiveRanges([self.live_range])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return self.symbolic_size

    def mark_allocated(self):
        assert not self.allocated
        self.allocated = True

    def finalize(self, pool, offset):
        assert self.pool is None and self.offset is None
        self.pool = pool
        self.offset = offset
        return self

    def codegen_alloc_from_pool(self, wrapper):
        assert self.pool
        node = self.node
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        return wrapper.codegen_alloc_from_pool(
            self.pool.name, self.offset, node.get_dtype(), shape, stride
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node={self.node.get_name()}, "
            f"live_range={self.live_range}, "
            f"size_hint={self.size_hint}, "
            f"symbolic_size={self.symbolic_size}, "
            f"pool={self.pool.name if self.pool else None}, "
            f"offset={self.offset})"
        )

# 教学源码：Empty 完整产品定义；本注释仅标识教学副本。
class Empty(AllocationTreeNode):
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """

    size_hint: int

    def get_live_ranges(self):
        return LiveRanges([])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return 0

    def is_empty(self):
        return True

# 教学源码：ClearCacheOnAllocateMixin 完整产品定义；本注释仅标识教学副本。
class ClearCacheOnAllocateMixin(MemorySplitProtocol):
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """

    def allocate(self, block: Allocation, is_last: bool):
        is_allocated = self._allocate(block, is_last)
        if is_allocated:
            self.clear_cache()
        return is_allocated

    def clear_cache(self):
        self.get_live_ranges.clear_cache(self)
        self.get_size_hint.clear_cache(self)
        self.get_symbolic_size.clear_cache(self)

# 教学源码：TemporalSplit 完整产品定义；本注释仅标识教学副本。
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """

    allocations: List[AllocationTreeNode]

    def _allocate(self, block: Allocation, is_last: bool):
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        block_live = block.get_live_ranges()
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        if len(overlapping) > 1:
            # TODO(jansel): we could try harder here by merging overlapping in space
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()

            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            if slot_size == block_size:
                # perfect fit
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(
                    SpatialSplit.create(block, slot_size - block_size)
                )
            else:  # grow this allocation
                assert is_last
                self.allocations = [
                    *(
                        SpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        return LiveRanges(
            itertools.chain.from_iterable(
                x.get_live_ranges().ranges for x in self.allocations
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        if not self.allocations:
            return 0
        return max(x.get_size_hint() for x in self.allocations)

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        if not self.allocations:
            return 0  # type: ignore[return-value]
        return sympy.Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self

# 教学源码：SpatialSplit 完整产品定义；本注释仅标识教学副本。
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """

    left: TemporalSplit
    right: TemporalSplit

    @staticmethod
    def create(left, extra_space):
        assert isinstance(left, AllocationTreeNode)
        assert isinstance(extra_space, int) and extra_space >= 1
        return SpatialSplit(TemporalSplit([left]), TemporalSplit([Empty(extra_space)]))

    def _allocate(self, block: Allocation, is_last: bool):
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    @cache_on_self
    def get_live_ranges(self):
        return LiveRanges(
            itertools.chain(
                self.left.get_live_ranges().ranges, self.right.get_live_ranges().ranges
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        return _align(self.left.get_size_hint()) + self.right.get_size_hint()

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()

    def finalize(self, pool, offset):
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )
        self.clear_cache()
        if self.right.is_empty():
            return self.left
        return self

# 教学源码：AllocationPool 完整产品定义；本注释仅标识教学副本。
class AllocationPool:
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """

    device: torch.device
    root: TemporalSplit
    can_expand: bool = True
    restrict_live_range: Optional[LiveRange] = None
    name: Optional[str] = None
    names_to_del: List[str] = dataclasses.field(default_factory=list)
    creation_cache: Dict[str, str] = dataclasses.field(default_factory=dict)

    def allocate(self, block: Allocation, is_last: bool):
        if self.restrict_live_range and not self.restrict_live_range.contains(
            block.live_range
        ):
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False

    def allocate_at_end(self, block):
        block.mark_allocated()
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
        return True

    def finalize(self, name):
        assert not self.name
        self.name = name
        self.names_to_del.append(name)
        self.root.finalize(self, 0)

    def codegen_create(self, wrapper, code: IndentedBuffer):
        assert self.name
        nbytes = self.root.get_symbolic_size()
        for block in self.root.allocations:
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                # optimization: fuse first allocation and pool creation
                node = block.node
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                self.creation_cache[block.codegen_alloc_from_pool(wrapper)] = self.name
                return
        else:
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )

    def codegen_destroy(self, wrapper, code: IndentedBuffer):
        code.writeline(wrapper.make_free_by_names(self.names_to_del))

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

# 教学源码：AllocationPools 完整产品定义；本注释仅标识教学副本。
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """

    device_to_pools: Dict[torch.device, List[AllocationPool]] = dataclasses.field(
        default_factory=dict
    )

    def get_pools(self, block):
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        pools = self.get_pools(block)

        for pool in pools:
            if pool.allocate(block, is_last=pool is pools[-1]):
                return

        # everything is full, make a new pool
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=config.memory_pool != "none",
            )
        )
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
        pools = self.get_pools(block)
        if pools and config.memory_pool in ("outputs", "combined"):
            pools[-1].allocate_at_end(block)
        else:
            # create a new pool
            block.mark_allocated()
            pools.append(
                AllocationPool(
                    block.device,
                    TemporalSplit([block]),
                    can_expand=config.memory_pool == "combined",
                )
            )

    def finalize(self):
        """Called at the end of allocation process"""
        for i, pool in enumerate(
            itertools.chain.from_iterable(self.device_to_pools.values())
        ):
            pool.finalize(f"pool{i}")

    def pprint(self):
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)
```

### 两种节点分别表示“同一位置换时间”和“同一时间换位置”

`TemporalSplit.allocations` 中的子节点不得有重叠 live range，它们共享同一块起始
offset，在不同时段轮流使用。它的容量是各子节点 size 的最大值，符号容量用
`sympy.Max` 表达。

`SpatialSplit(left,right)` 则让左右节点同时存在于不同地址。右侧 offset 是：

```text
parent_offset + align(left.symbolic_size)
```

当前 `align` 按 64 bytes 对齐。因此时间关系决定能否共址，空间关系决定必须相邻时
各自从哪里开始；allocation tree 同时保存这两种关系。

`ClearCacheOnAllocateMixin` 解释了为什么每次成功插入都要清除 live-ranges、size-hint
和 symbolic-size 缓存：树结构已经变化，旧聚合值不能继续用于下一次放置。最终
`finalize(pool,offset)` 沿树递归，把 pool 身份和 byte offset 写入每个 Allocation。

### 三块 allocation 的完整插入过程

设三块都在 CPU，单位为 bytes：

```text
A: size=64, live=[1,4)
B: size=32, live=[4,6)
C: size=48, live=[2,5)
```

先建立 pool：`root=TemporalSplit([A])`。A 已标记 allocated，slot size=64。

插入 B 时，`[1,4)` 与 `[4,6)` 在端点相接但不重叠。`TemporalSplit._allocate` 的
overlapping 列表为空；slot 64 大于 B 的 32，于是追加
`SpatialSplit.create(B,32)`。这里右侧 32 bytes 是 B 所在时段的未用空间。A 与 B 仍
位于同一 temporal slot 的 base offset 0：

```text
TemporalSplit
├─ A(64, [1,4))                         offset 0
└─ SpatialSplit                         同一时间槽的 64-byte 形状
   ├─ B(32, [4,6))                      offset 0
   └─ Empty(32)                         offset 32
```

插入 C 时，它与 A 在 `[2,4)` 重叠，也与 B 在 `[4,5)` 重叠。根节点发现两个
overlapping child，不能把 C 递归塞入其中任一个。若当前 pool 是可扩展的最后一个
候选，`AllocationPool.allocate_at_end` 建立新的 spatial 右支：

```text
TemporalSplit
└─ SpatialSplit
   ├─ 原 temporal slot，symbolic size=64     offset 0
   └─ TemporalSplit([C]), size=48           offset 64
```

finalize 后 A.offset=0、B.offset=0、C.offset=64；pool 总 symbolic size 为
`align(64)+48=112` bytes。A/B 因时间不重叠复用地址，C 因同时跨过两者的生存期必须
放到另一空间。这三个 offset 是算法推演；开头固定案例的两个 pool 各只有一个
8×8 float32 allocation，实际 offset 都是 0。

### perfect fit、局部递归和扩容

- 新块与一个 child 重叠时，根节点递归调用那个 child 的 `allocate`，尝试在其空间树
  内找位置；与多个 child 重叠时当前实现直接拒绝，没有跨多个空间节点重新拼接。
- 不重叠且 size 等于 slot，直接把 block 追加为另一个 temporal occupant。
- 不重叠且 size 小于 slot，用 `SpatialSplit(block,Empty)` 补齐 slot。
- 不重叠但 size 大于 slot，只有 `is_last=True` 才可扩容：旧 occupant 各自补上新增
  空间，再加入大 block；非最后 pool 返回 False，让调用方尝试后续 pool 或新建 pool。

`AllocationPools.allocate` 按 device 取 pool 列表，依次尝试。只有列表最后一个 pool
可以按配置扩展；全部失败就创建新 pool。输出走 `allocate_output`：根据
`memory_pool` 配置选择追加到末尾 pool，或建立不能随意扩展的独立 pool。这样返回值
所在的 pool 会一直保留到函数把结果交给调用者，不会因内部计算已经结束而提前删除。

### 从 byte offset 回到 tensor view

`Allocation.codegen_alloc_from_pool` 把最终 pool name、offset、dtype、shape、stride
交给 wrapper。pool 本身优先寻找“某个 allocation 恰好覆盖整个 pool”的情形，直接以
该 tensor 的 shape/stride 创建；否则申请一维 uint8、长度为 symbolic nbytes 的底层
storage。各 tensor 再基于对应 byte offset 建 view。

pool 会在第一条使用它的 allocation line 之前创建；最后一条内部使用结束后，wrapper
删除 `pool.name` 和累积的逻辑 tensor 名。究竟是哪两条 wrapper line 负责首用和末用，
不是 allocation tree 自己决定的，而是 K06 的 `mark_first_last_usage` 标在具体
`AllocFromPoolLine`/`DeallocFromPoolLine` 上。

### K05 迁移题

若 B 的 live range 改成 `[3,6)`，它与 A 在 `[3,4)` 重叠，不能再共享 offset 0；树会
为 B 建空间分支。若只把 B size 从 32 改成 96、live 仍为 `[4,6)`，时间上仍可复用，
但 slot 必须从 64 扩到 96；只有当前节点允许作为最后可扩展 pool 时才在原 pool 扩容。
若 C live 改成 `[6,8)`，它与 A、B 都不重叠，三块可以都进入同一 temporal slot，
最终容量取 `max(64,32,48)=64`，而非相加为 144。

K05 到此完成。K06 将保留剩余 25 个 pool planning 方法，沿 pass 顺序把原 wrapper
line 转成 group、live range、allocation tree、首末 pool 使用和最终代码。


## K06：MemoryPlanner 按固定 pass 把时间线变成 pool 代码

### 完整源码：group、pool line 与 planner passes

本次 planner 输入沿用开头的完整时间线：buf0 分配后被第一次 mm 写入，relu 后以 buf1 继续使用；buf2 由第二次 mm 写入并以 buf3 返回。inplace 关系先形成 G1={buf0,buf1} 与 G2={buf2,buf3}；G2 是输出。配置为 intermediates pool，timestep 从 0 开始，pools 为空。passes 依次把相关 line 换成 group line、得到 G1/G2 的 live range、按大小放入 pool、标记 pool 第一次创建与最后一次删除，再输出代码；因为 G2 是输出，它的生存期延长到末尾，不会在 return 前删除。

```python
# 教学源码：BufferGroup 完整产品定义；本注释仅标识教学副本。
class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: ir.Buffer):
        self.node = node
        self.names = [node.get_name()]
        self.is_output = False
        self.allocation: Optional[Allocation] = None
        self.live_range = LiveRange(float("inf"), -float("inf"))

    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
        self.live_range = LiveRange(
            min(timestep, self.live_range.begin),
            max(timestep, self.live_range.end),
        )

    def sym_nbytes(self):
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize

    def make_allocation(self):
        assert not self.allocation, "multiple allocations"
        assert isinstance(self.live_range.begin, int), "live ranges not computed"
        nbytes = self.sym_nbytes()
        # For now, fallback value will be used if we encounter an unbacked SymInt. The longer-term plan is to have
        # size_hint() use better heuristics for unbackeds, at which point the fallback value will be ignored.
        size_hint = V.graph.sizevars.size_hint(nbytes, fallback=64)
        self.allocation = Allocation(
            self.node,
            self.live_range,
            size_hint=size_hint,
            symbolic_size=nbytes,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, "
            f"live_range={self.live_range}"
        )

# 教学源码：PoolMemoryPlanningLine 完整产品定义；本注释仅标识教学副本。
class PoolMemoryPlanningLine(MemoryPlanningLine):
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""

    group: BufferGroup
    timestep: Optional[int] = None

    @property
    def node(self):
        return self.group.node

# 教学源码：AllocFromPoolLine 完整产品定义；本注释仅标识教学副本。
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        assert allocation and allocation.pool
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.extend(self.group.names)
        alloc_from_pool = allocation.codegen_alloc_from_pool(self.wrapper)
        if alloc_from_pool in pool.creation_cache:
            code.writeline(
                self.wrapper.make_tensor_alias(
                    name, pool.creation_cache[alloc_from_pool], "alloc"
                )
            )
        else:
            pool.creation_cache[alloc_from_pool] = name
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )

# 教学源码：DeallocFromPoolLine 完整产品定义；本注释仅标识教学副本。
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""

    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        if self.is_last_pool_usage:
            assert self.group.allocation and self.group.allocation.pool
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)

# 教学源码：MemoryPlanner 完整产品定义；本注释仅标识教学副本。
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: Optional[List[BufferGroup]] = None

    def plan(self, lines: List[Any]) -> List[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        # drop any removed buffers
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                assert name not in name_to_group
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                assert new_name not in name_to_group
                # TODO(jansel): we should support reusing buffers created via ExternKernelAlloc
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = set(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names)

        assert self.buffer_groups is None
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    lines[i] = AllocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, FreeIfNotReusedLine):
                assert not line.is_reused
                if line.node.get_name() in name_to_group:
                    lines[i] = DeallocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, ReuseLine):
                if line.node.get_name() in name_to_group:
                    line.delete_old = False

    def compute_live_ranges(self, lines):
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)
                        line.timestep = timestep
            else:
                worklist.popleft()

        timestep += 1
        assert self.buffer_groups is not None
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        assert config.memory_pool in ("none", "intermediates", "outputs", "combined")
        assert self.buffer_groups is not None

        for group in self.buffer_groups:
            group.make_allocation()

        outputs: List[Allocation] = []
        intermediates: List[Allocation] = []
        for group in self.buffer_groups:
            assert group.allocation
            if group.is_output and config.memory_pool != "combined":
                outputs.append(group.allocation)
            else:
                intermediates.append(group.allocation)

        for block in sorted(
            outputs,
            key=lambda x: (
                x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate_output(block)

        for block in sorted(
            intermediates,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate(block)

        self.pools.finalize()

    def mark_first_last_usage(self, lines):
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
        seen = set()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)

        seen = set()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_last_pool_usage = (
                        pool.root.get_live_ranges().end <= line.timestep
                    )
                    seen.add(pool)
```

### 五个顶层步骤及其两个内部子步骤不能交换

`MemoryPlanner.plan` 先复制 line 列表，随后固定执行：

```text
drop removed
→ compute groups 并 convert pool lines
→ compute live ranges
→ make Allocation 并放入 pools
→ 标记每个 pool 的第一次/最后一次使用
```

removed 必须先清掉，否则会制造不存在的 group；live range 必须在 line 转换后计算，
因为只有 `PoolMemoryPlanningLine` 携带 group；allocation tree 又依赖已经确定的 live
range；首末使用最终依赖 allocation 已经指向具体 pool。

这里顶层箭头确实是五步。若把第二步内部的 `compute_buffer_groups` 与第四步末尾的
`pools.finalize()` 也单独计数，才得到七项实际工作；它们不是另外两个平级 pass。

`compute_buffer_groups` 把 inplace 名合并并标记 output。`convert_to_pool_lines` 只替换
在映射中的 allocate/free，并让原 ReuseLine 不删除旧名。`compute_live_ranges` 使用 K04
的分段 timestep，末尾把 output 延长一个时刻。

### 从 live range 到具体 pool

`BufferGroup.make_allocation` 用 owning node 的 `storage_size * dtype.itemsize` 得到符号
byte 数；`size_hint(...,fallback=64)` 只为排序和树的静态近似，`symbolic_size` 才进入
运行时代码。`allocate_groups` 按配置区分输出和中间值：

- output 按 `(size_hint, -live_length)` 从小到大处理，并调用 `allocate_output`；
- intermediate 按 `(-size_hint, -live_length)` 从大到小处理，再走普通 pool allocate；
- `memory_pool=combined` 时 output 也可进入 intermediate 集合；其他配置保留相应边界。

先放大中间块，使后续小块更容易填入已有 temporal/spatial 空隙。输出单独处理是为了
让返回 storage 的销毁时机不被普通中间值绑住。全部放置后，`pools.finalize()` 依遍历
顺序命名为 pool0、pool1，并递归写入 offset。

固定案例中输出组 G2 先成为 `pool0`，中间组 G1 成为 `pool1`：

```text
pool1 / G1: buf0,buf1，live [1,4)
pool0 / G2: buf2,buf3，live [3,5)，is_output
```

名字编号表达 finalize 顺序，不表达执行时创建顺序。运行代码仍在第一次使用 G1 时先
创建 pool1；到第二次 mm 输出前才创建 pool0。

### 第一次创建与最后一次销毁

`mark_first_last_usage` 正向扫描 `AllocFromPoolLine`。某个 pool 首次出现时把该行的
`is_first_pool_usage=True`；之后同 pool 的 allocation 不重复创建底层 storage。

反向扫描 `DeallocFromPoolLine`，每个 pool 只考察最后出现的一条。只有
`pool.root.get_live_ranges().end <= line.timestep` 才标记 `is_last_pool_usage`：dealloc
line 必须已到达整个 pool 所有 occupant 的结束点，不能仅因当前 group 结束就销毁。

生成时，首个 Alloc line 先 `pool.codegen_create`。每次 allocation 把 group 的全部名字
加入 `names_to_del`，再生成 pool+offset 的 tensor view；若相同 view 表达式已在
`creation_cache` 中出现，改为 tensor alias，避免重复构造。最后 Dealloc line 调用
`codegen_destroy`，一次删除 pool 名与收集的逻辑名。

实验产物对应为：

```python
pool1 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
buf0 = pool1  # alloc
buf1 = buf0;  # reuse
...
pool0 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
buf2 = pool0  # alloc
...
del pool1, buf0, buf1
buf3 = buf2;  # reuse
return (buf3,)
```

pool1 在 G1 最后用户第二次 mm 之后删除；pool0 含输出，正文中不会在 return 前出现
对应销毁。这里实际观察到 group、首用创建、末用删除与 output 保活；两个 pool 都只有
offset 0，非零 offset 仍由 K05 的源码推演支持。

### removed、别名缓存和动态大小的条件对照

- planning 完成前 buf0 被加入 removed：相关 Allocate/Free/Reuse line 先变 Null，group
  和 pool 都不会为它保留幽灵容量。
- 两个逻辑 tensor 得到完全相同的 `codegen_alloc_from_pool` 表达式：第二个使用
  `make_tensor_alias` 指向 creation cache 中的第一个名字；storage/view 不重复创建。
- symbolic nbytes 无可靠 hint：排序使用 fallback 64，但 pool 的最终一维 uint8 长度和
  tensor offset 仍保留符号表达式；fallback 不是运行时分配容量。

### K06 迁移题

若 pool1 再加入 G3，G3 的 dealloc line 比 G1 晚，那么反向扫描遇到 G1 的 dealloc 时
即使 G1 自身结束，也可能因 pool aggregate end 更晚而不标记销毁；必须等 G3 对应的
末尾 dealloc。若 G3 是输出，aggregate live range 延长到规划末尾，pool 不能在 wrapper
内部删除。若配置从 `intermediates` 改成 `combined`，输出可以与兼容中间值进入同一
pool，但 output 延长后的 live range仍会阻止同时段地址覆盖。

至此，可以从一条 wrapper 时间线判断何时分配、何时释放、何时复用，以及 pool planner 怎样把多块逻辑 buffer 放入较大的 allocation pool。

## 从新时间线判断一次复用是否安全

读到一段新 wrapper 时，先确定本次走哪条路径。默认 planner 按 free/allocate 顺序
维护 LIFO 候选栈；pool planner 才先把已经确定的 inplace 名合并成 storage group，
再按普通调用切分 planning timestep。pool 中两个 group 只有 live ranges 不重叠时才
可进入同一 temporal slot；重叠时必须进入不同 spatial offset。随后再检查 device、
symbolic byte capacity、输出延长和特殊身份。两条路径都不能覆盖仍可被调用者或后续
kernel 观察的值。

例如新增 D=16 bytes、live=[5,7)：它与 A=[1,4) 不重叠，却与 B=[4,6) 在 [5,6)
重叠；所以不能只因“写在 A、B 后面”便断言 offset 0 可用。allocation tree 会沿实际
重叠 child 查找空间；若没有位置，再扩展最后 pool 或新建 pool。这正是从本篇机制判断
新情况，而不是记住 `pool0`、`buf1=buf0` 两个产物名字。




