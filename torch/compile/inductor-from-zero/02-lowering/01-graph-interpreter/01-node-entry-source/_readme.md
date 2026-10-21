# Q211 作者记录：GraphLowering 输入登记

> 第五轮状态：叶子完成。重新按五个完整方法、全部输入类别、部分状态失败、layout/device/alignment 与迁移题验收。

## 读者、层次与目标

- 所在层：深度 3，Q21 的第一个实现叶子。
- 读者前提：已从 Q21 区分 FX env、`graph_input_names`、`graph_inputs`、operations/buffers，并知道 TensorBox 是 lowering 中传递张量结果的外层对象。
- 本叶子职责：按 placeholder 顺序取得 example input，把 tensor 登记成带 layout 的 InputBuffer/TensorBox，把标量登记成 SymPy 表达式，并记录名字、设备和对齐信息。
- 入口：当前 FX placeholder、Interpreter 的 example input 迭代位置、GraphLowering 的 ShapeEnv/子图名与图状态。
- 正常出口：返回 TensorBox、SymPy Expr 或 None 给 FX env，并更新对应图级输入状态。
- 失败/后续边界：未知类型在 tensor assert 失败；bool 可在 placeholder 成功但在 output 合同失败；alias TODO、对齐运行时保护以及算子消费不属于本方法。
- 承诺深度：五个完整方法及 Tensor、动态 Tensor、符号/普通标量、None、BackwardState、未知类型、used/unused device 与限定名的具体状态。

## 声明范围与资料

- 产品文件：`/work/pytorch/torch/_inductor/graph.py`。
- 完整方法：`symbolic_sizes_strides`（261）、`static_sizes_strides`（298）、`qualify_name`（631）、`add_device_info`（717）、`placeholder`（922）。它们在类中不连续，正文按产品出现顺序排列，每个方法内部完整。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 独立实验：`../../../experiments/04_graph_lowering_nodes.py` 与 `graph_lowering_nodes.txt`；同时保留完整 compile 的逐节点观测。
- 不使用旧教程；`reuse_shape_env=False` 的新建符号来源分支仅按当前源码解释，实验覆盖复用 ShapeEnv。

## 篇内知识点队列

| 顺序·编号 | 问题、状态与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | placeholder 顺序如何取得 example 并先记录接口名 | scalar/tensor/未知 object | 完成并复读 |
| 2·K02 | 静态与动态 size/stride 怎样进入 FixedLayout | 连续、转置、`[u0,3]` | 完成并实测 |
| 3·K03 | TensorBox/InputBuffer 怎样进入两个输入表 | used tensor | 完成并实测 |
| 4·K04 | SymTypes、int/float/bool、None、BackwardState 的出口差异 | 六类 scalar | 完成并实测 |
| 5·K05 | 原 target 与限定名分别存在哪里 | `name='branch'` | 完成并实测 |
| 6·K06 | used/unused device 与 alignment 在何时登记 | 两张 CPU 输入图 | 完成并实测 |
| 7·K07 | 未知类型与后续 output 失败怎样保留部分状态 | object、bool | 完成并实测 |

当前点：K07 完成；篇内无剩余。根队列下一项 Q212。

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文位置 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| 静态连续 Tensor | FixedLayout `[2,3]/[3,1]` | 主例/stride | direct lowering | 叶子机制完成 |
| 静态非连续 Tensor | 保留 `[3,2]/[1,3]` | stride | transpose 实测 | 叶子机制完成 |
| 动态 Tensor、复用 env | 保留 SymPy `u0` | 动态分支 | `[u0,3]` 实测 | 叶子机制完成 |
| 动态 Tensor、新 env | ConstantSource 创建符号 | 动态分支 | 当前源码 | 源码机制完成，未单列运行 |
| SymInt | 取 `node.expr`，不建 buffer | 四类输入 | unbacked u0 实测 | 叶子机制完成 |
| int/float | sympify 后登记 | 状态表 | 3、2.5 实测 | 叶子机制完成 |
| bool | placeholder 登记 BooleanTrue；output 后续失败 | 四类输入后说明 | 实际错误及部分状态 | 叶子机制完成 |
| None | 只保留接口位置，返回 None | 四类输入 | 实测 NoneAsConstantBuffer | 叶子机制完成 |
| BackwardState | 同样返回 None、不登记 graph_inputs | 四类输入 | 实例实测 | 叶子机制完成 |
| 未知 object | 追加输入名后 tensor assert | 四类输入后说明 | 实际 AssertionError | 叶子机制完成 |
| 子图限定名 | 原名在 names，限定名在 input dict/buffer | 名字节 | branch_x 实测 | 叶子机制完成 |
| used tensor device | device type/mapping 登记 | 名字设备节 | CPU used 实测 | 叶子机制完成 |
| unused tensor device | InputBuffer 有，device state 空 | 名字设备节 | CPU unused 实测 | 叶子机制完成 |
| alignment | helper true 才加入限定名 | 名字设备节 | 当前 CPU false + 源码 | 边界完成 |

## 实验事实

标量与特殊输入：

```text
int 3      -> graph_inputs['scalar']=sympy.Integer(3)
float 2.5  -> sympy.Float
None       -> names 有 scalar，graph_inputs 无，output NoneAsConstantBuffer
BackwardState -> 同上
SymInt u0  -> graph_inputs 为 Symbol u0
bool True  -> placeholder 登记 BooleanTrue，output 阶段 AssertionError
object()   -> names 已追加，graph_inputs 空，placeholder Tensor assert
```

Tensor：

```text
contiguous: size=[2,3], stride=[3,1]
transpose:  size=[3,2], stride=[1,3]
dynamic:    size=[u0,3], stride=[3,1], symbolic=True
branch:     graph_input_names=['x'], keys=['branch_x']
used CPU:   device_types=['cpu'], mapping cpu->x
unused CPU: InputBuffer 仍有；device_types=[], mapping=[]
```

当前 CPU 样例 `aligned=False`。它只证明这些样例的 helper 结果；不能外推所有 CPU/GPU 对齐。完整 compile 的 x/y placeholder 也实测返回 `TensorBox→StorageBox→InputBuffer`，operations/buffers 仍为零，随后普通算子才建立计算描述。

## 源码完整性审计

- 五个方法分别从产品类中完整复制；方法内部语句、docstring、TODO 与长 alignment 注释均保留。
- symbolic 的 reuse/new-env 两分支、SymInt→expr 列表转换完整。
- placeholder 的所有类型分支、assert、静态/动态选择、qualify、两张表、users/device、alignment 和 return 完整。
- 教学注释统一 `# 教学：`；正文明确代码块去掉类级缩进，不改变 AST。

已将教学代码块重新包入临时类，与产品五个方法逐一做 AST 对照。源码无漏行不能代替上述类型、状态与失败出口验收。

## 只凭正文的作者理解验收

1. **为什么转置输入不能只登记 shape？** `[3,2]` 的 stride `[1,3]` 决定逻辑位置到存储位置；误写连续 stride 会读取不同元素。
2. **branch 子图中为什么 names 是 x、dict key 是 branch_x？** names 保存调用接口顺序和原 target；qualify 发生在 tensor 分支稍后，服务生成期全局唯一命名。
3. **unused CPU 输入为什么仍有 InputBuffer，却不在 device_types？** 创建 buffer 不以 users 为条件；add_device_info 被 `current_node.users` 条件保护。
4. **bool 的错误为什么不能归到 placeholder？** bool 分支已 sympify 并写 graph_inputs；实际异常来自后续 output 不接受 BooleanTrue。
5. **object 失败后为什么 names 已变化？** append 位于全部类型判断之前；assert 没有事务回滚。
6. **动态 `[u0,3]` 如何保持 u0？** example 标记 symbolic，进入 symbolic_sizes_strides；reuse env 分支转换现有 size/stride，不另造具体整数。
7. **None 与 BackwardState 如何影响接口？** 都占 graph_input_names 位置并返回 None，但不成为 graph_inputs 键或存储。
8. **graph_inputs_original 为什么保存 InputBuffer 而非 TensorBox？** 它保留输入的底层原始身份，后续 output/mutation 合同比较输入是否被替换或写回。
9. **样例 tensor 的数值是否编进 InputBuffer？** 不；登记 device/dtype/size/stride 和名字，运行时再由 wrapper 传实际存储。

答案均可由正文中的源码、具体状态和实验推出。实际读者反馈尚无。

## 本轮整改与续写

原稿源码完整度较好，但实验声明只覆盖 Tensor、正文未区分 bool placeholder 与 output 失败，也未实际验证 BackwardState、限定名、unused device、转置 stride 和动态 symbol。本轮扩展 direct-lowering 实验，按真实部分状态回修正文与覆盖表。

## 第五轮叶子验收

- **完整源码**：五个分散方法均按产品出现顺序完整保留；每个方法内部语句、TODO、长 alignment 注释和两条 symbolic 分支无省略。
- **分支与出口**：静态/动态 Tensor、SymTypes、int/float/bool、None、BackwardState、未知 object、used/unused input、子图限定名和 alignment 均有进入状态与出口。
- **具体状态**：连续与转置 tensor 给出 size/stride 和具体元素位置；`[u0,3]` 保留符号；Tensor+alpha 表逐步展示两张输入表与 env；object/bool 区分 placeholder 部分状态和后续 output 失败。
- **理解迁移**：正文能推出未使用 CPU tensor 仍建 InputBuffer 但不登记 device，子图 names/key 不同，以及同一 ShapeEnv 符号在多个维度保持同一表达式。
- **证据边界**：reuse ShapeEnv、特殊输入和 used/unused 路径实测；新 ShapeEnv 符号来源与 alignment true 仅按当前源码，不外推设备行为。

Q211 第五轮叶子复核完成。下一项 Q212：GraphLowering 的 get_attr 与 call_function 分派完整源码。
