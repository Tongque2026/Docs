# Pattern Matcher：从候选锚点到图替换

[返回图优化](../README.md) · [上一篇：三个处理阶段](../01-pass-stages/README.md) · [返回总览](../../README.md)

实现叶子：[候选扫描与应用完整源码](01-apply-source/README.md) · [replacement 嵌入完整源码](02-replace-source/README.md)

一张图里可能有很多可替换结构，Inductor 也可能已经注册了很多规则。逐条拿所有规则扫描所有节点会做大量无关比较。`PatternMatcherPass` 把一组规则保存在一个对象里，并按规则输出端节点的 `(op,target)` 分组；阶段代码调用这个对象的 `apply` 时，它先取可能相关的节点，再尝试对应规则。

先把本篇涉及的对象连起来：

```text
PatternMatcherPass 对象
├─ pass_name：用于标识本轮改图
└─ patterns：末端节点的 (op, target) → PatternEntry 列表
                                      │
阶段代码调用 apply(当前 FX 图)         │ 每个 entry 保存匹配结构、附加条件和命中动作
                 │                    │
                 └──── 找候选并逐条尝试 ┘
```

注册代码把匹配结构、附加条件和命中动作组成一种具体 `PatternEntry`，再调用
`entry.register(matcher)`。`register` 从匹配结构最外层取得 `(op,target)`，把 entry
追加到 `matcher.patterns[(op,target)]` 的列表。之后 pre-grad、joint 或 post-grad 的
阶段代码调用同一个 matcher 的 `apply(gm.graph)`。
`apply` 命中时直接修改传入的图，返回成功应用的 entry 数量；没有命中时返回 0，图保持原样。一个阶段可以调用多个 `PatternMatcherPass` 对象，一个对象也可以保存多条 entry。

本篇接下来用一条规则走完注册、阶段调用、匹配和替换。候选扫描循环和 replacement
嵌入的逐行实现分别放在两个实现叶子。

FX node 的 `op` 表示它是哪类节点，`target` 表示这类节点具体调用谁。例如 `a=torch.ops.aten.add.Tensor(x,y)` 对应 `op="call_function"`，意思是调用一个函数；它的 `target` 是 `aten.add.Tensor`，`args` 则引用 x 和 y。后文的 `(node.op, node.target)` 就是在检查这两项。

## 独立实验的规则

实验 [`experiments/02_pattern_matcher.py`](../../experiments/02_pattern_matcher.py) 先用
`matcher = PatternMatcherPass(pass_name="tutorial_distribute_mul")` 创建对象；这个名字只用于
给图变换日志标识这一轮 pass，不参与候选选择或规则匹配。随后建立下面的 `PatternExpr`
结构：

```text
mul.Tensor(
    add.Tensor(x, y),
    2
)
```

`KeywordArg("x")` 和 `KeywordArg("y")` 表示“此处可以是外部传入的对象，并以给定名字保存在匹配结果中”。它们把捕获值写入 `Match.kwargs`，所以本例得到具名的 x、y。另一种叶子 `Arg()` 不给捕获值命名，而是把它写入 `Match.args`；一棵 pattern 中有多个 `Arg()` 时，递归匹配按深度优先顺序把它们依次加入列表。本例没有使用 `Arg()`，所以 `match.args` 为空。替换图表达：

```text
x * 2 + y * 2
```

选择这条分配律规则，是因为一次命中会同时暴露新增节点、重接使用者和按剩余使用者删除旧节点三种变化。

实验使用 `register_graph_pattern` 注册。这个注册装饰器把上述 pattern、float32 附加条件
和 `distribute_mul` handler 组成一个 `GraphPatternEntry`，然后调用
`entry.register(matcher)`。根 pattern 是 `CallFunction(aten.mul.Tensor,...)`，所以
`register` 得到键 `("call_function", aten.mul.Tensor)`，将 entry 追加到对应列表。注册后
的具体状态是：

```text
matcher.patterns = {
  ("call_function", aten.mul.Tensor): [本例的 GraphPatternEntry]
}
```

handler 在规则命中时接收 `Match` 以及其中捕获的 x、y，再调用 replacement helper 把
上面的替换图嵌入当前图。本例 handler 没有 `return`，因此返回 `None`；
`GraphPatternEntry.apply` 也不读取这个返回值，改图结果完全通过原地副作用传递。注册装饰器
最后返回原 handler；长期留在 matcher 中的是 entry。

`_users=MULTIPLE` 是规则边界的一部分。若保留默认值 1，共享 add 会在结构匹配时因 user 数不符而拒绝，后文便观察不到“替换成功，但只删除已无使用者的旧节点”。

## 一条 Graph pattern 规则怎样完成 replacement

```text
PatternMatcherPass.apply(gm.graph)
   │
   ├─ 1. 按 (node.op, node.target) 找候选锚点
   ├─ 2. 从输出侧递归匹配 PatternExpr 结构
   ├─ 3. 检查 mutation region 和 extra_check
   ├─ 4. GraphPatternEntry.apply 调用 handler
   ├─ 5. handler 调用 replacement helper 嵌入替换图
   └─ 6. 图中的节点关系已经被修改；返回本次应用数量
```

这里的 `gm` 是当前阶段持有的 FX `GraphModule`，产品阶段把其中的 `gm.graph` 传给
`apply`。`apply` 也兼容直接传入 `GraphModule`，内部仍会取出 `.graph`；本篇沿产品阶段
路径统一写 `apply(gm.graph)`。图不是第二个返回值；新节点、新引用和被删除的旧节点都
直接留在 `gm.graph` 中。阶段代码接着对这张图运行后续 pass。

这里的 `PatternExpr` 是规则中的一个结构节点，例如外层 `CallFunction(mul, ...)`、内层 `CallFunction(add,...)` 和两个 `KeywordArg`；递归匹配就是把这些结构节点逐一对应到实际 FX node。mutation region 是按写入屏障分出的编号；add 在写入前、mul 在写入后时，两者编号不同，不能作为一个纯计算子图替换。

### 1. 用末端 target 缩小搜索范围

注册时，`PatternEntry.register` 已经把规则放入当前 `PatternMatcherPass.patterns` 中以
`(op,target)` 为键的列表。本例的表项是：

```text
("call_function", aten.mul.Tensor) → [这条 GraphPatternEntry]
```

阶段代码随后对这个对象调用 `apply`。它先取得图中的
`call_function/aten.mul.Tensor` 节点，而不是让每条规则从每个节点盲搜。

`apply` 逆序访问候选。以输出侧节点为锚点有两个好处：模式的内部输入已经在图中存在；一次替换擦除节点后，逆序遍历更容易避免重新处理刚插入的替换内部。

实验的 `relu(x+y)` 图没有 `aten.mul.Tensor`。registry 中虽有 mul bucket，`graph.find_nodes` 找不到根候选，最终应用次数为 0。这不同于“找到了 mul 节点，但它的输入结构不符合规则”：前者没有候选节点，后者有候选但匹配失败。

### 2. 递归结构匹配并收集 Match

从 `mul` 开始：第二个参数必须等于常数 2；第一个参数必须是 `aten.add.Tensor`；add 的两个参数分别由 `KeywordArg` 捕获为 x、y。

成功的 `Match` 不只有布尔值。它保存：

- `nodes`：本次模式包含的原图节点，本例为 add、mul；
- `kwargs`：捕获的 `x`、`y` FX 节点；
- `targets`：支持可变 target 模式时记录实际 target；
- `MatchContext.pattern_to_node`：每个 pattern expression 对应哪个实际节点。

如果同一个 PatternExpr 在模式中重复出现，`MatchContext` 会要求两处实际对象相同；这能表达“必须是同一输入”，而不是两个碰巧形状一样的输入。

实验让 `add(same,same)` 的两个参数引用同一个 `KeywordArg` 对象。图 `x+x` 命中；图 `x+y` 即使 x、y 都是 shape `[3]` 的 float32，也以 `repeated pattern differs` 失败。这里核对的是同一个 FX 对象，不是 shape 或 dtype 相同。

### 3. 候选和 Match 依次经过通用检查与规则条件

取得候选节点后，`apply` 先查看这个候选的 FakeTensor 输入和输出。例如普通 complex
tensor 目前不能由这条 codegen 路径读写，关闭 C++ codegen 时 CPU 输出也不能由它生成，
于是整个候选在尝试 entry 之前就被跳过，原节点随后仍由正常 Lowering/fallback 路径处理。

候选通过后，`entry.pattern.match(node)` 才返回完整 `Match` 或失败结果。得到 `Match` 后，
`apply` 检查 `Match.nodes` 是否跨越一次写入：跨越 mutation region 的匹配会丢弃，防止
规则把写入前后的计算误当成可自由重排的纯计算子图。这道检查通过后，才把同一个
`Match` 交给 entry 的 `extra_check`。

本实验的 `extra_check` 读取捕获节点 `x.meta["val"].dtype`：

| 请求 | 候选/结构 | mutation region | extra_check | 应用次数与出口 |
| --- | --- | --- | --- | --- |
| `relu(x+y)` | 没有 mul 锚点 | 不进入 | 不进入 | 0；图不变 |
| `(x+y)*3` | 有 mul；常数 `3 != 2` | 不进入 | 不进入 | 0；图不变 |
| float32 `(x+y)*2` | 成功 | add、mul 都为 0 | 成功 | 1；两个 mul 后接 add |
| int64 `(x+y)*2` | 成功 | add、mul 都为 0 | 失败 | 0；原 add 后接 mul |
| `a=x+y; copy_(x,y); a*2` | 成功 | add 为 0、mul 为 1 | 不进入 | 0；原图和写回保持 |

这五行形成一条可定位的拒绝链。没有替换不能笼统解释成“没匹配”：它可能没有根候选，可能在递归结构处失败，可能被通用副作用边界拦住，也可能建立完整 Match 后被规则自己的条件拒绝。

跨 mutation 的例子还说明为什么 region 判断不能省略。`a` 保存覆盖前的 `x+y`；若把它改写成位于 `copy_` 后面的 `x*2+y*2`，新表达式会读取已被覆盖的 x。实际观测为 add=0、copy_=1、mul=1，所以 pass 拒绝这次跨区匹配。

本实验的 `GraphPatternEntry.extra_check` 只检查 dtype。另一种注册入口
`register_replacement` 会把更复杂的 `check_fn` 保存为 entry 的 `extra_check`：通用
pattern 先忽略示例尺寸中的具体整数，以便找到相同结构；`check_fn` 再从已有 `Match`
捕获节点的 `meta["val"]` 取得真实 shape、stride、dtype、device 和 requires-grad，重新
trace 并匹配 specific pattern。通过后，它把按本次真实参数 trace 的 replacement graph
保存进这个 `Match`，供 `ReplacementPatternEntry.apply` 使用。这条支线不改变生命周期：
结构匹配先产生 Match，extra_check 再读取并可能补充它。

所有检查通过后，`apply` 把这个 Match 交给当前 entry。先比较三种 entry 怎样接住它。
另外两类 entry 的注册入口与本例采用同一组基本对象。`register_replacement` 接收 search
函数、replacement 函数、示例输入和目标 matcher；它 trace 搜索结构，并把 pattern、上文
的 `check_fn` 与捕获参数整理函数组成 `ReplacementPatternEntry`。
`register_lowering_pattern(pattern, extra_check, pass_dict=matcher)` 返回装饰器；装饰器把
pattern、条件和被装饰的 lowering handler 组成 `LoweringPatternEntry` 并注册到 matcher。
因此三类 entry 都长期保存在 bucket 中，差别发生在命中后怎样消费 Match：

| entry | 命中后的本层出口 |
| --- | --- |
| `ReplacementPatternEntry` | 从 Match 取得捕获参数和 `replacement_graph`，立即把替换图嵌入当前 FX 图 |
| `GraphPatternEntry` | 立即调用 handler：第一个参数是 `Match`；其余位置参数来自 `match.args`，具名捕获值来自 `match.kwargs`。handler 可执行任意声明的图手术，本例的 x、y 位于 `kwargs`，并在这里调用 replacement helper |
| `LoweringPatternEntry` | 在旧节点前插入一个 FX `call_function`；target 是已经绑定本次 Match 的 lowering handler，参数来自 Match 捕获值；随后重接旧节点 users 并删除无用旧节点。GraphLowering 以后读到这个 call 时才让 handler 建立 Inductor IR |

三类 entry 共用候选搜索、结构绑定和条件检查。本实验选择第二条
`GraphPatternEntry` 分支，但它的 handler 调用与第一条共用的 replacement helper，所以下面
继续追踪 helper 怎样嵌入 replacement graph。

三种 `entry.apply(...)` 都返回 `None`，把插入节点、重接 users 或调用 handler 的结果留在
当前图里；外层 `PatternMatcherPass.apply` 不读取 entry、handler 或 replacement helper 的
返回值来判断成功。所有检查通过后，它先把本地 `count` 加一，再调用 `entry.apply`；entry
正常结束后才更新两个全局统计量。若 entry 抛出异常，整个 pass 不会正常返回应用次数，
源码也没有在这里承诺回滚已经发生的局部改图。

### 4. Graph entry 的 handler 调用 replacement helper

实验预先用 `make_fx` 得到 replacement `GraphModule`。`GraphPatternEntry.apply` 把 Match
捕获的 x、y 作为参数调用 `distribute_mul` handler；handler 再调用
`match.replace_with_graph(replacement_graph, [x,y])`。此时输入状态是“原图中的旧
add/mul、捕获到的 x/y、以及一张描述 `x*2+y*2` 的替换图”；出口状态是“新节点已复制
进原图、旧 mul 的使用者改读新 add、没有使用者的旧节点被删除”。内部 `Replacer` 完成
这次复制：replacement 的 placeholder 绑定到原图中的 x、y，每个 `call_function` 都在
旧输出之前建立一个对应的新节点。helper 返回 `None`，结果留在原地修改后的当前图中。

然后它处理出口：

```text
旧 mul 的 users ───────┐
                      ▼
旧 add → 旧 mul      新 add（来自 x*2 + y*2）
   删除旧 mul，再在无 user 时删除旧 add
```

新输出接管旧输出的全部 users。若新节点自身没有 `val` meta，替换逻辑从旧输出复制 meta。多输出规则还会处理旧 packed return 的 `getitem` users，将它们分别接到新输出。

最后 `match.erase_nodes()` 逆序删除已无用户的旧匹配节点。它不会强删仍有外部 users 的节点。共享实验实际得到：

```text
改写前：x,y → add ─┬→ mul(*2) ─→ tuple[0]
                    └→ relu    ─→ tuple[1]

改写后：x,y → add ─────────────→ relu ─→ tuple[1]
         x → mul(*2) ─┐
                       ├→ new_add ─────→ tuple[0]
         y → mul(*2) ─┘
```

应用次数为 1：旧 mul 的 tuple user 改读 `new_add`，旧 mul 删除；旧 add 仍有 relu user，因此保留。规则只声明接管旧 mul 的输出，所以 relu 不会被偷偷改成读取 replacement 的结果。

多输出 replacement 会按旧 `getitem` 的整数索引，把每个使用者接到对应新输出。完整的 packed-output 分支在 [replacement 嵌入实现](02-replace-source/README.md)展开。

### 5. 交出修改后的节点关系

所有检查通过、准备调用 entry 时，pass 先把局部 `count` 加一。entry 正常结束后：

- `pattern_matcher_count` 加一；
- `pattern_matcher_nodes` 增加本次匹配节点数。

应用结束后，pass 返回本轮正常应用数量；`apply` 本身不调用阶段级的 `recompile()` 或
`lint()`。阶段入口继续在同一张图上运行其余 pass，随后由
[图状态维护](../03-graph-invariants/README.md)同步顺序、节点属性和可执行 `forward`
并检查结构。

## 用新情况检查理解

若把共享例中 pattern 的 `_users=MULTIPLE` 去掉，结果不再是“替换后删不掉 add”，而是在结构匹配时就因 add 有两个 users 被拒绝。user 约束决定能否形成 Match；`erase_nodes` 决定替换后哪些旧节点还能删除，这是两个不同时间点。

若把 `copy_` 移到 mul 之后，add 和 mul 会落在同一 region，mutation-region 这道门不再构成拒绝理由；是否应用还要继续看结构与 dtype。这个判断来自检查顺序和节点状态，不能只看整张图里“出现过 copy_”。

