# Pattern Matcher：从候选锚点到图替换

[返回图优化](../README.md) · [上一篇：三个处理阶段](../01-pass-stages/README.md) · [返回总览](../../README.md)

实现叶子：[候选扫描与应用完整源码](01-apply-source/README.md) · [replacement 嵌入完整源码](02-replace-source/README.md)

Pattern Matcher 要完成的不是“看到两个算子名字就融合”，而是一次有约束的图变换：先找可能的末端节点，再核对整个子图结构、捕获输入、检查真实 shape/dtype 等附加条件，最后替换节点并维护使用关系。

FX node 的 `op` 表示它是哪类节点，`target` 表示这类节点具体调用谁。例如 `a=torch.ops.aten.add.Tensor(x,y)` 对应 `op="call_function"`，意思是调用一个函数；它的 `target` 是 `aten.add.Tensor`，`args` 则引用 x 和 y。后文的 `(node.op, node.target)` 就是在检查这两项。

改图以后，内存中的 Graph 已经变化，但 `GraphModule.forward` 仍可能是修改前生成的 Python 代码。`recompile()` 根据新 Graph 重建 `forward`；`lint()` 检查输入先于使用者、节点属于当前图等结构条件，发现错误就报错。结果无人引用且删除不会丢掉写入等副作用的节点可由 dead code elimination 删除，后文简称 DCE。

## 独立实验的规则

实验 [`experiments/02_pattern_matcher.py`](../../experiments/02_pattern_matcher.py) 注册下面的模式：

```text
mul.Tensor(
    add.Tensor(x, y),
    2
)
```

`KeywordArg("x")` 和 `KeywordArg("y")` 表示“此处可以是外部传入的对象，并以给定名字交给匹配成功后的回调函数”。本篇源码也把这个回调叫 handler；它与 Codegen 中“把 load/add/store 翻译成目标代码”的 ops handler 不是同一种对象。实验实现还给内部 add 设置 `_users=MULTIPLE`，允许它除了 mul 之外另有使用者。规则另有一个条件：`x.dtype` 必须是 `float32`。替换图表达：

```text
x * 2 + y * 2
```

这个分配律变换只用于观察 matcher；它不是在声称 Inductor 产品优化应该把一项加法和一项乘法改成两项乘法和一项加法。

`_users=MULTIPLE` 是规则边界的一部分。若保留默认值 1，共享 add 会在结构匹配时因 user 数不符而拒绝，后文便观察不到“替换成功，但只删除已无使用者的旧节点”。

## 一次应用经过五步

```text
PatternMatcherPass.apply(gm)
   │
   ├─ 1. 按 (node.op, node.target) 找候选锚点
   ├─ 2. 从输出侧递归匹配 PatternExpr 结构
   ├─ 3. 检查 mutation region 和 extra_check
   ├─ 4. PatternEntry.apply 执行替换
   └─ 5. 更新计数；阶段外围再排序、更新 meta、lint/recompile
```

这里的 `PatternExpr` 是规则中的一个结构节点，例如外层 `CallFunction(mul, ...)`、内层 `CallFunction(add,...)` 和两个 `KeywordArg`；递归匹配就是把这些结构节点逐一对应到实际 FX node。mutation region 是按写入屏障分出的编号；add 在写入前、mul 在写入后时，两者编号不同，不能作为一个纯计算子图替换。

### 1. 用末端 target 缩小搜索范围

注册时，`PatternEntry.register` 把规则放入以 `(op,target)` 为键的表。本例根模式是 `CallFunction(aten.mul.Tensor, ...)`，因此 pass 先取得图中的 `call_function/aten.mul.Tensor` 节点，而不是让每条规则从每个节点盲搜。

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

### 3. 结构相同以后，再检查 dtype 和写入位置

当前 `PatternMatcherPass.apply` 在调用 `extra_check` 前还有两道通用检查。第一道查看候选节点的 FakeTensor 输入和输出；例如普通 complex tensor 目前不能由这条 codegen 路径读写，关闭 C++ codegen 时 CPU 输出也不能由它生成，于是跳过这个 pattern，原节点随后仍由正常 Lowering/fallback 路径处理。第二道检查匹配到的节点是否跨越一次写入：跨越 mutation region 的匹配会丢弃，防止规则把写入前后的计算误当成可自由重排的纯计算子图。

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

### 4. 把 replacement graph 嵌入原图

实验预先用 `make_fx` 得到 replacement `GraphModule`。handler 调用 `match.replace_with_graph(replacement_graph, [x,y])`。此时输入状态是“原图中的旧 add/mul、捕获到的 x/y、以及一张描述 `x*2+y*2` 的替换图”；出口状态是“新节点已复制进原图、旧 mul 的使用者改读新 add、没有使用者的旧节点被删除”。内部 `Replacer` 完成这次复制：replacement 的 placeholder 绑定到原图中的 x、y，每个 `call_function` 都在旧输出之前建立一个对应的新节点。

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

多输出 replacement 还多一层图手术。若旧节点返回 packed tuple，而外部通过 `getitem(old,0)`、`getitem(old,1)` 使用，replacement 返回新节点列表时，`replace_with_graph` 读取每个 getitem 的整数索引，把它的 users 重接到对应新节点，再删除旧 getitem 和 packed 节点。遇到非 getitem user 会触发 `can't handle`，不会猜测拆包语义。

### 5. 计数和阶段收尾各证明什么

每次真正应用后：

- pass 的局部返回值加一；
- `pattern_matcher_count` 加一；
- `pattern_matcher_nodes` 增加本次匹配节点数。

这些计数证明规则被应用及覆盖多少匹配节点，但不单独证明替换数值正确。实验另做了三项检查：`gm.graph.lint()`、`gm.recompile()`、替换后输出与原函数 `torch.testing.assert_close`。

真实 pre/joint/post-grad 阶段还会在合适时机调用稳定拓扑、fake meta 更新、DCE 或其他图维护。Pattern entry 只负责它声明的局部变换，不承担整个阶段全部不变量。

## 三种 entry 的出口不同

当前实现用共同的结构匹配入口承载三类处理：

| entry | 命中后做什么 | 典型用途 |
| --- | --- | --- |
| `ReplacementPatternEntry` | 把追踪得到的 replacement graph 嵌入 FX 图 | 一组 FX 计算换成另一组 FX 计算 |
| `GraphPatternEntry` | 调用 handler，由 handler 自行改图 | 需要定制节点删除、重接等操作；也可用一组示例输入 trace 出替换图再嵌入，源码称 `replace_by_example` |
| `LoweringPatternEntry` | 插入一个稍后执行 handler 的 FX call | 命中子图后直接在 Lowering 时构造 Inductor IR |

第三类说明 pattern matcher 不只服务 FX-to-FX replacement；有些 post-grad 模式会延迟到 Lowering handler。但它仍先在 FX 图层识别结构。

三类 entry 共用的是“怎样得到 Match”，图出口却不同。Replacement entry 使用 `extra_check` 已准备的 `match.replacement_graph`；Graph entry 在旧输出前调用自定义 handler，由 handler 决定图手术；Lowering entry 新建包住 handler 的 `call_function`，复制旧输出 meta、重接全部 users，再删除可删旧节点。因此只知道命中次数，还不能判断图会变成哪一种形式。

## 为什么还要具体 shape 重新匹配

`register_replacement` 可以先用 example inputs 生成通用 pattern。注册时若把示例中的尺寸 32 直接写成模式必须等于 32，尺寸 64 的同结构计算就会被错误拒绝；因此初次模式可以忽略这类具体整数。真正命中用户图后，`check_fn` 从捕获节点的 `meta["val"]` 取得实际 shape、stride 和 dtype，重新 trace 并匹配更具体的 pattern，再运行 `extra_check`。只有通过后，才为这次真实参数 trace replacement graph。

这形成两级判断：通用结构负责快速找候选，具体重匹配负责确认当前输入确实满足规则。

具体过程是：按 search function 的参数顺序，从第一次 Match 的命名捕获中取得 FX 节点；读取各节点 `meta["val"]`，以真实 size、stride、dtype、device 和 requires-grad 重建示例输入；重新 trace search function 并在原输出节点上匹配。specific pattern 与用户 `extra_check` 都成功后，才按这组真实参数 trace replacement，并保存到 `match.replacement_graph`。因此 replacement 可以随这次真实形状生成，并非永远复用注册时的一张固定图。

## 用新情况检查理解

若把共享例中 pattern 的 `_users=MULTIPLE` 去掉，结果不再是“替换后删不掉 add”，而是在结构匹配时就因 add 有两个 users 被拒绝。user 约束决定能否形成 Match；`erase_nodes` 决定替换后哪些旧节点还能删除，这是两个不同时间点。

若把 `copy_` 移到 mul 之后，add 和 mul 会落在同一 region，mutation-region 这道门不再构成拒绝理由；是否应用还要继续看结构与 dtype。这个判断来自检查顺序和节点状态，不能只看整张图里“出现过 copy_”。
