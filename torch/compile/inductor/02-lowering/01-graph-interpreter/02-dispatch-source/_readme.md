# Q212 作者记录：GraphLowering 算子与常量分派

> 第五轮状态：叶子完成。重新以八个完整方法、选择顺序、常量状态、missing/fallback 出口和证据分层验收。

## 读者、层次与目标

- 所在层：深度 3，Q21 的第二个实现叶子。
- 读者前提：Q211 已把 placeholder 变成 TensorBox/SymPy/None，FX Interpreter 会把 env 中结果转换成当前 call_function 的 args。
- 本叶子职责：把函数 target 分派给 container 操作、pattern lowering、已注册 lowering、动态 fallback 或明确错误；把模块属性分派成内联常量、ConstantBuffer、Subgraph 或 TorchBindObject。
- 入口：call_function 的 target/IR args/kwargs，或 get_attr 的点分 target；同时读取 lowering registry、配置与 GraphModule 属性。
- 正常出口：可放回 FX env 的 IR/容器值；常量路径还更新 constants/reprs/name tables。
- 失败出口：两种 MissingOperator、LoweringException、未知 callable assert、call_module/call_method assert，以及属性非支持类型 assert。
- 承诺深度：八个完整方法、全部选择顺序、常量命名/去重和实际分派状态；被调用 lowering 内部留在所属模块。

## 声明范围与资料

- 产品文件：`/work/pytorch/torch/_inductor/graph.py`。
- 完整方法：`allocate_non_dup_const_name`、`add_tensor_constant`、`constant_name`、`call_function`、`can_inline_constant`、`get_attr`、`call_module`、`call_method`，按产品出现顺序排列。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 实验：扩展后的 `../../../experiments/04_graph_lowering_nodes.py` 与产物；Q24 的特殊路径实验只作为真实 fallback IR 的相邻证据。
- 未运行：TorchBind ScriptObject、自定义白名单 op、CUDA 跨设备 constant copy；对应结论限于当前源码。

## 篇内知识点队列

| 顺序·编号 | 问题、状态与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | container getitem 与 pattern passthrough 为什么先于 registry | tuple、标记 callable | 完成并实测 |
| 2·K02 | 已注册 lowering 如何统一调用并包装错误 | add、故意 ValueError | 完成并实测 |
| 3·K03 | 未注册 OpOverload 如何按 allowlist/config/decomp 分四路 | addcdiv、linalg_eig | 完成并实测主要三路；白名单源码 |
| 4·K04 | get_attr 类型优先级与强制保留配置 | 0d、小向量、表、子图 | 完成并实测 |
| 5·K05 | 常量名字、存储位置去重和冲突后缀 | view/offset/clone/非法字符 | 完成并实测 |
| 6·K06 | 跨设备常量、ScriptObject 和普通模块/方法边界 | 源码分支、assert probe | 完成并标明证据边界 |

当前点：K06 完成；篇内无剩余。根队列下一项 Q213。

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文位置 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| Python container getitem | 父 Interpreter 直接取项 | 两入口/实际 probe | tuple→20 | 叶子机制完成 |
| pattern passthrough | 标记 callable 直接执行 | 未注册表前 | 实际返回 tuple | 叶子机制完成 |
| 已注册 ATen lowering | registry 调用，返回 IR | add 案例 | compile 逐节点事件 | 叶子机制完成 |
| lowering 抛异常 | 包成 LoweringException 并保留 traceback | 未注册出口后 | ValueError probe | 叶子机制完成 |
| fallback allowlist | make_fallback 后统一调用 | 四出口表 | 当前源码 | 源码机制完成，未单列运行 |
| implicit fallback | 日志类型按 decomp，注册并统一调用 | 四出口表 | mock registration probe | 接口机制完成 |
| 有 decomp、禁止 fallback | MissingOperatorWithDecomp | 四出口表 | addcdiv 实测 | 叶子机制完成 |
| 无 decomp、禁止 fallback | MissingOperatorWithoutDecomp | 四出口表 | linalg_eig 实测 | 叶子机制完成 |
| 未标记普通 callable | OpOverload assert | 迁移题 | 当前源码 | 叶子机制完成 |
| GraphModule attr | Subgraph | 属性表 | 实际 branch | 叶子机制完成 |
| ScriptObject attr | TorchBindObject+表 | 属性表 | 当前源码 | 源码机制完成，环境未构造 |
| 0d Tensor | Constant，除非强制保留 | 属性表 | 默认/forced 实测 | 叶子机制完成 |
| 1d 长度≤8 | tensor lowering 内联 IR | 属性表 | 三项向量 Pointwise | 叶子机制完成 |
| 其他 Tensor | ConstantBuffer+三张表 | 属性表 | 3×4 表实测 | 叶子机制完成 |
| 同 storage/offset | 复用已有名 | 名字节 | 同 view 实测 | 叶子机制完成 |
| storage offset 不同 | 不去重 | 名字节 | 两个 view 实测 | 叶子机制完成 |
| 数值相同独立 storage | 不去重 | 名字节 | clone 实测 | 叶子机制完成 |
| 跨设备 constant | disable modes、to、重新登记 | 名字节 | 当前源码 | 源码机制完成，未跑 CUDA |
| call_module/call_method | AssertionError | probe | 两项实测 | 叶子机制完成 |

## 实验事实与边界

```text
get_attr:
  scalar -> Constant
  [1,2,3] -> TensorBox/StorageBox/Pointwise
  3x4 table -> TensorBox/StorageBox/ConstantBuffer
  branch GraphModule -> Subgraph
  forced keep -> scalar/small 均 ConstantBuffer

naming:
  first=layer_weight
  same position=layer_weight
  different offset=layer_weight_0
  independent equal clone=layer_weight_1
  1.bad-name=constant_1_bad_name

dispatch:
  container getitem=20
  marked passthrough=('passthrough',7)
  addcdiv -> MissingOperatorWithDecomp
  linalg_eig -> MissingOperatorWithoutDecomp
  implicit linalg_eig -> make_fallback once, then fallback-result
  failing lowering -> LoweringException, message starts ValueError
  call_module/call_method -> AssertionError
```

implicit fallback probe mock 的是 `make_fallback` 所登记的具体 lowering，以隔离 `call_function` 的控制流和统一调用点；它不证明真实 linalg_eig fallback 可执行。get_attr 真实调用当前 constant/tensor lowering。ScriptObject、allowlist、跨设备仅有源码证据。

## 源码完整性审计

- 八个方法全部保留，教学代码块去掉类级缩进后逐方法 AST 与产品相等。
- 常量去重的 mkldnn、size/stride/dtype/device/data_ptr/offset 条件、名称清理/冲突循环及三表写入完整。
- call_function 的 getitem、marker、四种 missing/fallback、统一 try/except 完整。
- get_attr 的 GraphModule、ScriptObject、强制保留、0d、小向量、普通 tensor 顺序完整。
- call_module/method 两个失败方法完整，原注释和 TODO 未删。

源码等价不能替代每个出口的状态与证据边界；mock/源码/真实 lowering 已分别记录。

## 只凭正文的作者理解验收

1. **为什么 tuple getitem 不查 lowerings？** 第一个分支识别 Python 容器并交父 Interpreter 执行，结果本身不是张量 IR 操作。
2. **有 decomposition 为什么仍抛 MissingOperatorWithDecomp？** AOTAutograd 分解阶段已过；禁用 implicit fallback 时 GraphLowering 不回退重跑分解。
3. **implicit fallback 后为什么仍进入 try/except？** make_fallback 将 target 注册进同一 lowerings 表；控制流落到统一 registry 调用点。
4. **0d 常量何时不是 Constant？** runtime folding、always keep 或 unsupported output 任一条件先命中，就在 shape 判断前建立 ConstantBuffer。
5. **两个值相等 tensor 为什么不一定去重？** 条件比较存储 data pointer 与 offset，不比较逐元素数值；独立 clone 的 pointer 不同。
6. **同 storage 不同 offset 为什么不去重？** offset 是显式必要条件，两个 view 指向不同起点。
7. **为什么小向量是 Pointwise 而不是 Python list？** get_attr 将 list 交给 tensor lowering，它构造可在生成代码中内联的 TensorBox 计算描述。
8. **pattern callable 为什么绕过 OpOverload assert？** `_inductor_lowering_function` 检查位于 registry/missing 分支之前，直接执行。
9. **call_module 为什么不能自动读取模块？** 进入 GraphLowering 前普通模块调用应已规范化；本层明确以 assert 保护此前提，只有 get_attr 的 GraphModule 作为控制流子图保留。

答案均可由正文和案例推出。实际读者反馈尚无。

## 本轮整改与续写

原稿源码和逻辑表较完整，但几乎所有常量分支与 missing/fallback 只靠推演；作者记录没有区分 mock、当前源码和真实 IR 的证明范围。本轮扩展实验并据结果补齐模块计划、处理覆盖和迁移验收。

## 第五轮叶子验收

- **完整源码**：八个分散方法逐一完整保留；常量去重、命名、跨设备复制、call_function 全部分派和 get_attr 类型优先级均无省略。
- **分支与出口**：container getitem、pattern callable、已注册 lowering、allowlist/implicit fallback、两类 MissingOperator、LoweringException、GraphModule/ScriptObject/三类 Tensor、call_module/method assert 全覆盖。
- **具体状态**：add 的 IR 参数与返回、四种未知 op 出口、0d/小向量/表/子图属性表，以及同 storage 同 offset、不同 offset、独立 clone 的命名状态均走到 env 或错误出口。
- **理解迁移**：正文能判断长度 9 常量进入 ConstantBuffer、IR getitem 不走 Python container 快速路、普通 callable 在 OpOverload assert 失败、数值相同独立 tensor 不去重。
- **证据边界**：真实 get_attr 与错误分支、mock fallback 控制流、Q24 真实 fallback IR、仅源码的 ScriptObject/CUDA 分别登记，未混为同一次实测。

Q212 第五轮叶子复核完成。下一项 Q213：GraphLowering output 与输入 mutation 合同完整源码。
