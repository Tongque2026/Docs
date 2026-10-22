# Q122 作者记录：`ReplacementPatternEntry`

> 第五轮状态：叶子完成。已重新按声明范围、所有输出形态、失败出口、具体图状态与迁移题验收。

## 读者、层次与完成目标

- 所在层：深度 3，Q12 的第二个实现叶子；与 Q121 一起完成 matcher 的扫描和替换实现。
- 读者前提：已学会 Q121 的候选、Match、entry 调用和 counters；知道 FX Node 的 users 与拓扑顺序。
- 本叶子职责：解释 replacement GraphModule，在原 graph 新建 call_function，按输出合同迁移 users/meta/tags，清理旧节点。
- 入口：成功 Match、待修改的原 Graph、replacement GraphModule，以及按 placeholder 排列的原图 args。
- 正常出口：新节点接入，旧输出及无用户 match 节点删除，原 Graph 保持可 lint/执行。
- 失败出口：unsupported replacement op、输出数量断言、packed 非 getitem user/错误索引、裸 Graph 的当前 Interpreter 构造限制等；源码不做事务回滚。
- 承诺深度：类装饰器到 apply 末尾的全部源码、所有分支与循环、单/多/packed/None 输出、两类 meta 和 tag 传播状态。

## 声明范围与资料

- 产品文件：`/work/pytorch/torch/_inductor/pattern_matcher.py:1029-1192`。
- 完整范围：`@dataclasses.dataclass class ReplacementPatternEntry`、字段 `normalize_args`、静态 `replace_with_graph` 及实例 `apply`。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 资料：当前源码、Match 的 `output_nodes/erase_nodes/replace_by_example` 当前实现、独立实验 `../../../experiments/02_pattern_matcher.py` 及产物。
- 原稿漏掉类装饰器，故此前完整源码验收无效；本轮已恢复并重新做 AST 对照。

## 篇内知识点队列

| 顺序·编号 | 问题、状态与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | Replacer 怎样绑定 placeholder、在原图新建 call_function、返回 output | 单输出 distribute mul | 完成并复读 |
| 2·K02 | 如何选最早插入点并按位置迁移多个旧输出 | 两旧/两新实验 | 完成并复读 |
| 3·K03 | packed tuple 怎样通过 getitem 拆接，失败时发生什么 | packed 成功与直接 user 失败 | 完成并复读 |
| 4·K04 | old/new 的 Node/None 组合及数量不等决策 | 两个 optional 实验 | 完成并复读 |
| 5·K05 | replacement meta、旧 meta、recompute/ac_graph_id 各怎样传播 | traced meta、手工无 meta、两节点标签链 | 完成并复读 |
| 6·K06 | apply 怎样检查 replacement graph 并规范化参数 | apply 末段源码与参数流 | 完成并复读 |

当前点：K06 完成；篇内无剩余项。根队列下一项为同深度 Q131。

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文位置 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| placeholder/output | Interpreter env 绑定/返回 | “Replacer”与单输出表 | 主替换实验 | 叶子机制完成 |
| call_function | 在原 graph 插入节点 | 单输出与多输出 | 前后图、call 顺序 | 叶子机制完成 |
| call_method/module/get_attr | run_node 抛 NotImplemented | “Replacer 只重建” | 当前源码 | 源码机制完成，未单独运行 |
| 裸 Graph 参数 | 当前 Interpreter 缺 named_modules | “Replacer 只重建” | 实际 AttributeError | 当前行为完成 |
| 单旧/单新 | users→new，擦除 old | 第一条路径 | distribute mul | 叶子机制完成 |
| 多旧/等量新 | 最早旧输出作插入点，按列表位置 zip | 两输出 | 实际结果与顺序 | 叶子机制完成 |
| old/new 都 None | 无动作 | None 表 | optional 实验 | 叶子机制完成 |
| old Node/new None | users 改 None、删 old | None 表 | removed 实验 | 叶子机制完成 |
| old None/new Node | assert | None 表 | 源码推演 | 叶子机制完成 |
| packed 单旧/多新 | getitem index 逐项改接 | packed 节 | 实际无 getitem/pack | 叶子机制完成 |
| packed 非 getitem user | `can't handle`，无回滚 | packed 节 | 实际异常 | 叶子机制完成 |
| replacement trace meta | Replacer 复制 val/tensor_meta | meta 节 | 三新节点实测 | 叶子机制完成 |
| 新输出无 val | 复制对应 old 全 meta | meta 节 | `old_only=17` | 叶子机制完成 |
| recompute/ac_graph_id | 从对应新输出反向至 args 前 | tag 节 | 三节点标签表 | 叶子机制完成 |
| match 内剩余节点 | 输出迁移成功后 erase_nodes | 单/packed路径 | Probe erase call + 主图 | 叶子机制完成 |
| apply | normalize args 后转静态方法 | 末节 | 当前源码 | 叶子机制完成 |

## 实验结果与证据边界

Q121 主替换已经实际覆盖单输出路径和共享旧 add 保留。本轮继续扩展同一实验：

```text
two old/two new: result ([0,6],[2,5])
new call order: add_tensor, mul_tensor, add_tensor_1
tags: first add/mul 都为 r1/g1；second add 无；captured input 无
packed: result ([5],[6])，最终无 pack_pair/getitem
packed direct user: AssertionError("can't handle")
new node without val: old_only=17、recompute=r2 从旧输出复制
[old,None] -> [new,None]: ([3],None)
[old0,old1] -> [None,new1]: (None,[5])
raw replacement Graph: AttributeError("Graph ... named_modules")
```

ProbeMatch 只实现产品静态方法实际调用的 `output_nodes()` 与 `erase_nodes()`，用于隔离 graph surgery；节点创建、users 重接、erase、meta 和标签逻辑均由产品 `replace_with_graph` 执行。它不能证明完整 PatternExpr 捕获过程，该部分由 Q12/Q121 的真实 matcher 案例覆盖。

packed 失败案例会在修改部分 getitem 后遇到 output user；只记录异常与非事务性，不把失败后的图当成合法产物。性能与不同设备不属于这段纯 FX graph surgery 的结论。

## 源码完整性审计

- 原稿遗漏的 `@dataclasses.dataclass` 已补回；类字段和两个方法签名完整。
- Replacer 三个禁用方法字段、run_node 三类路径、meta 分支与失败出口完整。
- output_nodes、单/多插入点、indices/min、percolate queue/visited/input_stops 完整。
- replacement Node 包装、maybe_getitem、replace 的 None/Node/Sequence 全部分支与产品长注释完整。
- 等量 zip、packed assert、with 块结束后的 `match.erase_nodes` 及 apply 的 assert/normalize 调用完整。
- 教学注释统一 `# 教学：`，无省略产品语句或注释。

每次正文改动后用 AST 比较产品类与教学代码块；完整性只证明源码范围，机制完成另由覆盖表与状态推演支持。

## 只凭正文的作者理解验收

1. **为什么两个新输出都插在更早旧输出前，却仍按 O0→N0/O1→N1？** 插入点按原图位置保证拓扑；zip 按接口列表位置决定对应，两种顺序职责不同。
2. **为什么 replacement placeholder 不产生新 placeholder？** Interpreter.run 用传入 args 初始化 placeholder env；后续新 call 直接引用这些原图 Node。
3. **packed 的 getitem0 怎样消失？** 取出 idx=0，递归 replace(getitem0,new[0])，迁移其 users 并擦除 getitem0；所有 users 完成后擦除 packed。
4. **packed 有直接 output user 时为什么不能保持原图不变？** 函数逐 user 原地修改且无事务；先前 getitem 可能已替换，遇到非 getitem 才抛错。
5. **replacement 已有 val 时，旧输出的 `old_only` 是否自动整体复制？** 不会；整体 old.meta 后备只在新节点无 val 时执行，recompute/ac_graph_id 另行显式传播。
6. **为何标签到 add 和 mul，却不到 x？** 反向队列访问新内部节点；x 属于 set(args) 的 input_stops，既不写标签也不继续越过。
7. **old=None/new=Node 会怎样？** 第一个分支断言 new 也必须 None，因此失败；可选接口不能凭 replacement 单方面新增原 pattern 未声明的输出。
8. **为什么裸 Graph 注解不能证明当前可运行？** 实际 Replacer/Interpreter 构造会访问 named_modules；实验对裸 Graph 得到 AttributeError，当前正常路径需 GraphModule。

答案均可从正文状态和源码推出。实际读者反馈尚无；作者复读不替代读者理解。

## 本轮整改与续写

原稿有单输出和 packed 文字，但漏掉 dataclass 装饰器，且未实际覆盖多输出插入点、None 分支、packed 失败、meta 两来源、标签停止或裸 Graph 边界。作者记录也不具备层次、入口/出口、完整覆盖与证据限制。本轮先扩展实验，再据当前结果重建全文。

## 第五轮叶子验收

- **完整源码**：类装饰器、字段、内部 Replacer、辅助函数、所有产品注释和 `apply` 均在连续范围内，教学注释有固定标记；重新确认没有用简化实现替代产品逻辑。
- **分支与出口**：单/多输出、packed、四种 old/new Node/None 组合、call_function 与不支持 op、裸 Graph 边界、输出数量/索引断言、meta 两来源、tag 传播和最终 erase 全覆盖。
- **具体状态**：单输出六时刻表、多输出插入点与位置映射、packed 两个 getitem 的逐次擦除、失败后的部分修改、tag 队列到 `input_stops` 停止均可逐步追踪。
- **理解迁移**：正文能推出 O1 更早时插入点选 O1 但接口仍 O0→N0/O1→N1；N0 的标签不会自动传到独立 N1；old=None/new=Node 会在第一分支断言。
- **证据边界**：ProbeMatch 只隔离 graph surgery；PatternExpr 捕获由 Q12/Q121 覆盖。packed 失败只证明非事务异常，不把残缺图当合法出口。

Q122 第五轮叶子复核完成。全系列下一项 Q131：stable topological sort 的完整源码与具体 pending/ready/waiting 状态。

## 第一次隔离冷读整改

- 报告：`../../../../inductor-fresh-reader-review/59_REPLACE_GRAPH_COLD_REVIEW_1.md`，评级 C。
- 分支虽齐全，但读者无法独立推出 Interpreter env、单输出 users、packed 失败残图、tag queue 和 normalize 参数映射。
- 本轮增加六节点 env 表，逐步展示 placeholder 绑定、fetch 参数替换、新节点写回 env 和 output 返回；增加单输出 users 变化；固定实验 users 顺序并画出异常后的部分图；展开 tag 的 queue/visited/input_stops；补 kwargs 捕获重排到 placeholder 顺序的例子和 traced meta 最终对照。

## 第二次隔离冷读验收

- 报告：`../../../../inductor-fresh-reader-review/60_REPLACE_GRAPH_COLD_REVIEW_2.md`，评级 A。
- 读者可独立推演 env、users、插入点、输出映射、packed 失败残图、meta/tag、normalize 与最终清理。
- 必须整改项：无。
- 当前状态：Q122 通过隔离冷读验收；下一篇为同层 Q131。
