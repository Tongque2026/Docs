# FX Graph Cache 查找源码：key 找同类图，guard 选择当前可用版本

[返回缓存、编译与加载](../README.md) · [上一叶子：Wrapper 复用规划](../../03-wrapper-memory/01-planner-source/README.md)

跨进程第二次编译能命中，要求缓存既不能把同一种计算错分成不同 key，也不能把当前
输入不适用的机器码误当成命中。本叶子把这两个问题分开：hash details 找“同类编译
请求”，每个候选保存的 guards 再判断“当前符号值能否使用这一版本”。

真正的上游入口是 `_compile_fx_inner`。非 AOT 的零算子图在这里直接返回 boxed
`gm.forward`。这里的 boxed 表示调用者把这张图的全部实参装在一个 Python list 中
传入，callee 接收这一个 list，而不是接收展开的 `fn(*args)`；零算子路径会先给普通
`gm.forward` 加上这层调用适配。它不会进入 FX Graph Cache；AOT mode 也绕过这条 cache 分支，但 AOT 的
零算子图不会走前述提前返回，而会继续自己的 codegen。只有 cache 开启、未被
`force_disable_caches` 禁用且不是 AOT mode 的普通单图编译，才调用
`FxGraphCache.load`。后文先展开 `load` 的完整控制顺序，再在结尾接回
`_compile_fx_inner` 给 cache 返回对象标记同一 boxed 调用协议，并交给 AOTAutograd 的
compiler plumbing：

```text
输入：compile_fx_fn、gm、example_inputs、fx_kwargs、inputs_to_check、local/remote
  |
  ├─ 前置检查或 key 序列化无法安全完成
  |      -> bypass：直接 compile，不查找、不保存
  |
  └─ 形成 key
         -> 依次检查 local 候选，再检查至多一个 remote 候选
              |
              ├─ 某候选 guard 对本次整数 hints 为真
              |      -> 恢复 wrapper callable
              |      -> 用真实 SymInt 再求 guard，安装到当前 ShapeEnv
              |      -> hit
              |
              └─ local/remote 候选读取或反序列化失败时跳过；或全部 guard 拒绝；
                 或选中 artifact 的 loader 抛 OSError
                     -> miss：调用 compile_fx_fn
                     -> 保存去掉 callable 的 metadata 副本

hit / miss / bypass
  -> 共同执行 post_compile（当前输入 stride、cudagraph、realign）
  -> load 返回 current_callable 非空的 CompiledFxGraph
  -> _compile_fx_inner 设置 _boxed_call=True，再交给上层 compiler plumbing
```

这个容错范围不能扩大到整个候选处理过程。整数 hint 的 guard 求值、wrapper artifact
重写、非 `OSError` 的 loader 异常、真实 SymInt 的第二次 guard 求值、统计恢复或 trace
记录失败都会继续向外传播。在 `_lookup_graph` 内，只有源码明确捕获的 local/remote
读取反序列化异常和 `load_by_key_path` 的 `OSError` 会退化为继续候选或 miss。
`load` 外层另行处理 remote client 构造失败、`BypassFxGraphCache` 和保存阶段的部分
pickle/write 异常，不能把这些不同作用域的捕获混成一条规则。

因此 bypass 与 miss 不同：miss 已经有安全 key，只是没有可用候选；bypass 连安全查找
的前提都不成立。hit 也不是直接返回磁盘对象，必须先恢复当前进程的 callable，再走
共同后处理。

贯穿全文的实测跨进程案例使用静态输入，不和后面的动态双 guard 推演混用：

```python
def fn(x):
    return torch.relu(x + 1) * 2

x = torch.arange(-3.0, 16.0)  # 19 项：-3,-2,...,15
```

run1 与 run2 各在独立进程中编译同一函数，并共享一个空的临时 cache 目录。run1 形成
key K，本地目录为空，所以 compile 得到原 graph B；保存副本清空 callable。这个静态
案例不需要用 `s0%16` 一类动态 guard 解释命中。run1 执行时，-3、-2、-1 三项经
`x+1` 和 relu 后都是 0，其余 0..15 变成 1..16；因此结果和为
`2*(1+2+...+16)=2*136=272`。run2 在新进程再次形成 K，从 artifact 恢复本进程的
`module.call`，同样得到 272。

后文的 `s0=32/19`、H0/H1 是独立的动态 shape 推演，用来解释“同 key 多候选怎样靠
guard 选择”，不是实验 15 的 run1/run2。动态候选若本次 hint 不满足 H0，就继续其他
叶子；全部拒绝才重新编译并保存新叶子。

## K01：把会改变代码生成的信息变成稳定 hash details

以 `relu(x+1)*2` 为例，FX 图结构相同仍不足以决定缓存安全。key 对象还保存：

```text
gm                         FX GraphModule 的规范化结构与属性
example_inputs             dtype/device、静态 shape/stride、动态维的符号身份
fx_kwargs                  排序后的编译参数，排除不稳定 graph_id
inputs_to_check            需要 alignment 检查的输入位置
deterministic settings     是否启用确定性算法及填充行为
CUDA matmul settings       TF32 和低精度 reduction 开关
torch_version              当前源码/版本 hash
system_info                系统与编译器环境
inductor_config            可移植配置快照
```

输入维度要分成三类信息：

| 信息 | 例子 | 进入 key 还是 guard |
| --- | --- | --- |
| 静态 tensor metadata | float32、CPU、静态维 32、stride 1 | 进入 key；改变后通常是另一类请求 |
| 动态维的符号身份 | 规范化图中的 `s0` | 规范化后的名字进入 key，使同一动态图归到一起 |
| 本次运行的整数 hint | 这次 `s0=16`，下次 `s0=19` | 不进入 SymInt 的 key reducer；用于逐候选求 guard |

所以 `s0` 的 hint 从 16 变成 19 时 key 可以不变，却可能从“16 的倍数版本”切到
尾部版本。若符号身份或静态 metadata 改变，则不能套用这条结论。

kwargs 先按名字排序；值若是 set，再排序后放入 `OrderedSetHolder`。因此调用者仅改变
字典插入顺序或 set 遍历顺序，不会无意义地改变 key；真正的配置值改变仍进入序列化结果。

### Tensor、FakeTensor 和 SymInt 保存的内容不同

自定义 pickler 不直接序列化容易跨进程变化的 Python 对象：

- FakeTensor 变成精简后的 `TensorMetadata`；普通情况下 storage_offset 清零、
  storage_bytes 去除，静态标记 tensor 才保留相应信息。
- 真 Tensor 被视为 GraphModule 常量，除 metadata 外还保存 `tolist()` 的值。两个图
  即使常量 shape/dtype 相同，常量 1 与常量 2 也不能共用被内联的代码。
- backed SymInt 只保存符号名字，如 `s0`，不保存当前 hint 19。当前值是否适用由候选
  guard 判断，因此同类动态图可以共享 key 下的多个版本。

metadata 中的 `torch.device` 还按等价值归一到同一对象。Python pickle 会记对象引用；
若两个等价 device 有时是同一对象、有时只是值相等，未经归一化会产生不同 byte stream。

### 从 details 到 key

进入后面的长源码前，先固定主要参数和值。类型标注 `Dict[str,Any]` 表示字符串键到任意配置值的字典；本例 `fx_kwargs` 至少含当前编译模式和 shape 相关选项。`gm` 是 `relu(x+1)*2` 的 FX GraphModule，`example_inputs=[x]`，x 是 float32 CPU tensor；`inputs_to_check` 是需要对齐检查的输入位置集合。序列化器把这些字段按稳定顺序写成字节，再由 hash 函数得到 graph key；加载阶段用该 key 找候选列表，再把本次符号值代入每个候选 guard。

`FxGraphCachePickler.dumps(details)` 开启 fast 模式，减少字符串 intern 对 byte stream 的
影响；`get_hash` 对结果做 SHA-256。`compiled_fx_graph_hash` 再加前缀 `f`，将该命名
空间与 codecache 中其他 artifact 区分。debug lines 为每个字段单独计算 hash 并输出
可读内容，它用于解释 key 差异，不参与最终 key 的再次组合。

```text
FxGraphHashDetails
   └─ custom pickle bytes
        └─ sha256
             └─ "f" + digest = fx graph key
```

### bypass 是 key 无法安全形成，不是 cache miss

真实常量若为当前不支持 pickle 的 mkldnn tensor，配置中含无法 pickle 的 callable，或
遇到显式不支持 reducer 的对象，代码抛出 `BypassFxGraphCache`。`load` 捕获后会调用
完整 compile function，但不会把这次情况记为普通 miss/save；因为系统连安全稳定的 key
都无法承诺。

### K01 条件推演

1. x 的具体数据从随机值 A 换成 B，但它只是运行时输入 FakeTensor：metadata 不含数值，
   key 不因数据内容改变；dtype、device、静态 shape/stride 改变则可能改变 key。动态维
   当前 hint 改变按第 3 项处理，不能笼统算作“shape 改变就换 key”。
2. GraphModule 属性中的常量从 1 改成 2：真 Tensor reducer 包含 values，key 改变。
3. backed `s0` 的 hint 从 16 改成 19：SymInt reducer仍只写 `"s0"`，key 可以不变；
   `s0 % 16 == 0` 候选的 guard 对 19 为假，查找继续其他候选或 miss。
4. `fx_kwargs` 中仅 `graph_id` 改变：该字段被明确排除，不改变 key；影响代码生成的其他
   配置改变则进入排序后的 kwargs 或 portable config。

K01 到此完成。K02 将打开 local/remote 候选迭代与 guard 求值，代入同一 key 下
`s0%16==0` 和尾部版本两个候选。


## K02：完整源码中的候选顺序与 guard 选择

### 完整源码：key 规范化、hash details 与 FxGraphCache

这段源码很长，第一次阅读只跟一条 local-hit 路径。当前规范化 FX 图与输入 metadata 算出的 key 记为 K；本地目录 K 下有两个候选文件 H0、H1，远端缓存关闭。ShapeEnv 给出 backed symbol 的当前 hint `s0=19`。H0 保存 guard `s0%16==0`，求值为 false；H1 保存 `s0%16!=0`，求值为 true，并记录 wrapper 的 artifact key P。查找 H1 后，loader 从 P 对应的 `.py` 恢复 `module.call`，把 H1 的 guard 安装回当前 ShapeEnv，然后返回 hit。前半的 pickler、hash details 和路径 helper 都是在建立 K 与定位 K/H0、K/H1；遇到与这条路径无关的 remote/save 分支可先看其输入输出注释，第二遍再代入 miss。

```python
# 教学源码：TOP._ident 完整产品定义；本注释仅标识教学副本。
def _ident(x: T) -> T:
    return x

# 教学源码：TOP.extract_tensor_metadata_for_cache_key 完整产品定义；本注释仅标识教学副本。
def extract_tensor_metadata_for_cache_key(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> TensorMetadata:
    """
    Extracts the tensor metadata and removes fields of the TensorMetadata
    that are not needed for caching
    """
    meta = extract_tensor_metadata(t)
    if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)

    # The pickle implementation avoids serializing the same object more than once.
    # That behavior means the byte stream we create to hash will vary if, for example,
    # we see two tensor objects with the same device, but the torch.device object is
    # actually the same object vs. merely equivalent. We want to produce the same hash
    # value in either situation, so we memoize the device objects and always reference
    # the same object for a given device. It's possible other metadata fields deserve
    # the same treatment, but so far we've only observed this issue with the device.
    if meta.device not in device_map:
        device_map[meta.device] = meta.device
    meta = dataclasses.replace(meta, device=device_map[meta.device])

    return meta

# 教学源码：TOP._reduce_fake_tensor 完整产品定义；本注释仅标识教学副本。
def _reduce_fake_tensor(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> Tuple[Callable[[T], T], Tuple[TensorMetadata]]:
    """
    See FxGraphCachePickler. Custom reducer to pickle FakeTensors.
    """
    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    return (_ident, (metadata,))

# 教学源码：TOP._reduce_tensor 完整产品定义；本注释仅标识教学副本。
def _reduce_tensor(
    device_map: Dict[torch.device, torch.device], t: Tensor
) -> Tuple[Callable[[T], T], Tuple[TensorMetadataAndValues]]:
    """
    See FxGraphCachePickler. Custom reducer to pickle Tensors.
    If we see tensors, we know they're constants stored as attributes on
    the GraphModule. Include the values in the key calculation. Small
    tensors will be inlined, so we can't serve the same cache entry for
    different values anyway. Large constants are treated as parameters,
    so we could conceivably reuse a cache entry. To do that, however,
    PyCodeCache would need more complexity to create a new module from its
    cache, but with the right constants attached as attributes.
    """
    if t.is_mkldnn:
        # TODO: These tensors don't currently pickle, so we can't cache a
        # compiled graph containing them. Just fail now. If mkldnn tensors
        # get pickling support, we can remove this.
        raise BypassFxGraphCache("mkldnn tensors unpickleable.")

    # Very large tensors could be expensive to copy to cpu and hash. Let's
    # at least report if we find slowness.
    start = time()
    values = t.tolist()
    elapsed = time() - start
    if elapsed > 1.0:
        warnings.warn(
            f"FX graph cache handling of a large constant took {elapsed:.1}s. Please file an issue."
        )

    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    return (_ident, (TensorMetadataAndValues(metadata, values),))

# 教学源码：TOP._reduce_symint 完整产品定义；本注释仅标识教学副本。
def _reduce_symint(s: SymInt) -> Tuple[Callable[[T], T], Tuple[str]]:
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # For hashing purposes, we only care about the name of the symbol and
    # not the backed value. We evaluate guards stored with a cached graph
    # to ensure a cached entity with SymInt args is safe to reuse.
    return (_ident, (str(s),))

# 教学源码：TOP._reduce_unsupported 完整产品定义；本注释仅标识教学副本。
def _reduce_unsupported(s: Any) -> NoReturn:
    """
    See FxGraphCachePickler. Custom reducer to handle any objects that we don't
    support and therefore raise to bypass caching.
    """
    raise BypassFxGraphCache("Reduce unsupported.")

# 教学源码：CLASS.FxGraphCachePickler 完整产品定义；本注释仅标识教学副本。
class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    # See extract_tensor_metadata_for_cache_key. Whenever we extract metadata during
    # pickling, we make sure devices always reference the same torch.device object.
    _device_map: Dict[torch.device, torch.device] = {}

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = functools.partial(_reduce_fake_tensor, _device_map)
    dispatch_table[torch.Tensor] = functools.partial(_reduce_tensor, _device_map)
    dispatch_table[torch.SymInt] = _reduce_symint
    dispatch_table[
        torch.fx.experimental._backward_state.BackwardState
    ] = _reduce_unsupported

    @classmethod
    def dumps(cls, obj: Any) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = cls(stream)
            # TODO: pickler.fast is technically deprecated. Will this work on new python versions?
            pickler.fast = True  # Run with pickler.fast so it doesn't intern strings, making the hash result more predictable
            try:
                pickler.dump(obj)
            except (TypeError, AttributeError) as e:
                # Some configs options are callables, e.g., post_grad_custom_pre_pass,
                # and may not pickle.
                log.warning("Can't pickle", exc_info=True)
                raise BypassFxGraphCache("Config options may be unpickleable.") from e
            return stream.getvalue()

    @classmethod
    def get_hash(cls, obj: Any) -> str:
        """
        Serialize an object using the FxGraphCachePickler and return a hash
        of the pickled object.
        """
        serialized_data = cls.dumps(obj)
        return sha256_hash(serialized_data)

    @classmethod
    def debug_lines(cls, inp: FxGraphHashDetails) -> List[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj: Any) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata_for_cache_key(cls._device_map, obj))
            elif isinstance(obj, bytes):
                return "<bytes>"
            elif type(obj) in cls.dispatch_table:
                # Run the reducer on the object
                return str(cls.dispatch_table[type(obj)](obj)[1])
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(inp).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = cls.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = cls.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = cls.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return lines

# 教学源码：CLASS.FxGraphHashDetails 完整产品定义；本注释仅标识教学副本。
class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """

    # Excluded kwargs param that are not stable between runs
    EXCLUDED_KWARGS = ["graph_id"]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
        inputs_to_check: Sequence[int],
    ) -> None:
        self.gm = gm
        self.example_inputs = example_inputs

        # Order kwargs so hashing is stable to changes in kwarg order.
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]

        # Alignment checks
        self.inputs_to_check = inputs_to_check

        # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        self.deterministic_algorithms_settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch_key()
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config_portable()

    def debug_lines(self) -> List[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """
        return FxGraphCachePickler.debug_lines(self)

# 教学源码：TOP.compiled_fx_graph_hash 完整产品定义；本注释仅标识教学副本。
def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fx_kwargs: Dict[str, Any],
    inputs_to_check: Sequence[int],
) -> Tuple[str, List[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + FxGraphCachePickler.get_hash(details)
    debug_lines = details.debug_lines()
    debug_str = "\n".join(debug_lines)
    log.debug(f"FX graph cache hash details for key {key}:\n{debug_str}")  # noqa: G004
    return key, debug_lines

# 教学源码：CLASS.FxGraphCache 完整产品定义；本注释仅标识教学副本。
class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    # TODO(masnesral): Investigate whether it's beneficial to store compiled graphs
    # in an in-memory cache after loading from disk.
    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "fxgraph")

    @staticmethod
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def _filter_backed_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt) and has_hint(s)]

    @staticmethod
    def _get_shape_env() -> Optional[ShapeEnv]:
        """
        Helper to get the shape env from the tracing context.
        """
        ctx = torch._guards.TracingContext.try_get()
        if not ctx:
            return None
        return ctx.fake_mode.shape_env

    @staticmethod
    def _lookup_graph(
        key: str,
        example_inputs: List[torch.Tensor],
        local: bool,
        remote_cache: Optional[RemoteCache[JsonDataTy]],
    ) -> Optional[CompiledFxGraph]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        """
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = FxGraphCache._filter_backed_symints(example_inputs)
        hints = [hint_int(s) for s in symints]

        def iterate_over_candidates() -> Generator[CompiledFxGraph, None, None]:
            if local:
                subdir = FxGraphCache._get_tmp_dir_for_key(key)
                if os.path.exists(subdir):
                    for path in sorted(os.listdir(subdir)):
                        try:
                            with open(os.path.join(subdir, path), "rb") as f:
                                yield pickle.load(f)
                        except Exception:
                            log.warning(
                                "fx graph cache unable to load compiled graph",
                                exc_info=True,
                            )

            if remote_cache:
                try:
                    if (cache_data := remote_cache.get(key)) is not None:
                        assert isinstance(cache_data, dict)
                        data = cache_data["data"]
                        assert isinstance(data, (str, bytes))
                        content = base64.b64decode(data)
                        yield pickle.loads(content)
                except Exception:
                    log.warning(
                        "fx graph cache unable to load compiled graph", exc_info=True
                    )

        # Iterate over any entries in the subdir for this key and evaluate
        # their guards to determine whether there's a hit.
        graph = None

        for candidate in iterate_over_candidates():
            if not candidate.guards_expr:
                # No guards to evaluate, so this is a hit.
                graph = candidate
                break

            # Evaluate the guard expression in the current context.
            # If there's not a cache hit, we don't want the evaluation to
            # affect the current env, e.g., cause the creation of new guards,
            # so we evaluate with the hints instead of the symbols.
            hit = bool(
                shape_env.evaluate_guards_expression(candidate.guards_expr, hints)
            )
            log.debug(
                "fx graph cache key %s evaluating guards [%s] with values %s => hit=%s",
                key,
                candidate.guards_expr,
                hints,
                hit,
            )
            if hit:
                graph = candidate
                break

        if graph is None:
            return None

        # See _save_graph(); we don't store the callable in the cache entry so
        # recreate it here from the PyCodeCache disk cache.
        artifact_path = get_path(graph.cache_key, "py")[2]
        code = graph.source_code
        if not os.path.exists(artifact_path):
            counters["inductor"]["fxgraph_lookup_write_file"] += 1
            Path(os.path.dirname(artifact_path)).mkdir(parents=True, exist_ok=True)
            cpp_pp = cpp_prefix_path()
            if os.path.basename(cpp_pp) in code:
                if cpp_pp in code:
                    # Great the name is correct
                    pass
                else:
                    # Old dir name is included, replace it
                    pattern = rf'#include\s*"[^"]+{os.path.basename(cpp_pp)}"'
                    code = re.sub(pattern, f'#include "{cpp_pp}"', code)

            write_atomic(artifact_path, code, make_dirs=True)

        try:
            graph.current_callable = PyCodeCache.load_by_key_path(
                graph.cache_key,
                artifact_path,
                graph.cache_linemap,
                graph.constants,
            ).call
        except OSError:
            # Not expected, but in case the PyCodeCache entry is removed from
            # underneath us, treat it as a cache miss and recompile.
            log.error("Failed to load cached artifact: %s", artifact_path)
            return None

        # Now re-evaluate with the symints to add any guards to the current env.
        if graph.guards_expr:
            check = bool(
                shape_env.evaluate_guards_expression(graph.guards_expr, symints)
            )
            assert check is True
            log.debug(
                "fx graph cache key %s post-load guards: %s", key, shape_env.guards
            )

        # Increment the cached metrics/counters by the amounts recorded when the FX
        # graph was compiled for this cache entry. Pretending these counters
        # were incremented normally is useful for testing with the cache enabled.
        metrics.CachedMetricsHelper.apply_deltas(graph.metrics_deltas)
        counters["inductor"] += graph.counter_deltas

        from .graph import GraphLowering

        GraphLowering.save_output_code(code)
        output_code_log.debug("Output code written to: %s", artifact_path)
        output_code_log.debug("Output code: \n%s", code)
        # On cache hit, use artifact path as filename
        trace_structured(
            "inductor_output_code",
            lambda: {"filename": artifact_path},
            payload_fn=lambda: code,
        )
        return graph

    @staticmethod
    def post_compile(
        compiled_graph: CompiledFxGraph,
        example_inputs: List[torch.Tensor],
        cudagraphs: BoxedBool,
    ) -> CompiledFxGraph:
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
        set_tracing_context_output_strides(example_inputs, compiled_graph)

        if cudagraphs:
            # It's possible that cudagraphs is enabled, but was disabled
            # during a previous compilation we're loading from the cache.
            # If so, we need to disable it on this new process too.
            if compiled_graph.disabled_cudagraphs_reason:
                if "cuda" in compiled_graph.device_types:
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {compiled_graph.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                cudagraph_post_compile(
                    example_inputs,
                    compiled_graph,
                    cudagraphs,
                )
        inputs_to_check = compiled_graph.inputs_to_check
        # cudagraphs could have been disabled from the earlier conditions
        # so we still need to realign inputs if that happens
        maybe_realign_inputs(
            cudagraphs,
            compiled_graph,
            inputs_to_check,
        )

        return compiled_graph

    @staticmethod
    def _save_graph(
        key: str,
        compiled_graph: CompiledFxGraph,
        example_inputs: List[torch.Tensor],
        local: bool,
        remote_cache: Optional[RemoteCache[JsonDataTy]],
    ) -> None:
        """
        Store a serialized CompiledFxGraph on disk.
        """
        disk_compiled_graph = copy(compiled_graph)
        # We can't really serialize callables that may be C++/Triton/etc.,
        # so we serialize their PyCodeCache disk cache location instead.
        # TODO: This could be better if we're ever able to serialize compiled
        # models to disk.
        disk_compiled_graph.current_callable = None

        # Before serializing, compute the guard expression that will be used to
        # ensure that a CompiledFxGraph is valid when loaded from the cache. It's
        # sufficient to consider only the SymInt args to the fx graph since the
        # Tensor shapes are already captured in the hash for the cache key. Any
        # Tensor arg with a symbolic shape will have a SymInt arg for the graph.
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None
        symints = FxGraphCache._filter_backed_symints(example_inputs)
        guards = shape_env.get_pruned_guards(symints)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(
            placeholders=symints, guards=guards
        )

        try:
            content = pickle.dumps(disk_compiled_graph)
        except Exception:
            log.warning(
                "fx graph cache unable to serialize compiled graph", exc_info=True
            )
            counters["inductor"]["fxgraph_cache_pickle_error"] += 1
            return

        try:
            if local:
                subdir = FxGraphCache._get_tmp_dir_for_key(key)
                if not os.path.exists(subdir):
                    os.makedirs(subdir, exist_ok=True)

                # Use a hash of the serialized CompiledFxGraph to get a unique file
                # name. The specific name doesn't matter since a lookup involves
                # iterating over all entries in the parent subdir.
                path = os.path.join(subdir, sha256_hash(content))
                write_atomic(path, content, make_dirs=True)

            if remote_cache:
                time_taken_ms = int((disk_compiled_graph._time_taken_ns or 0) // 1e6)
                cache_data: JsonDataTy = {
                    "data": base64.b64encode(content).decode("ascii"),
                    "time_taken_ms": time_taken_ms,
                }
                remote_cache.put(key, cache_data)
        except Exception:
            log.warning("fx graph unable to write to cache", exc_info=True)
            counters["inductor"]["fxgraph_cache_write_error"] += 1

    @staticmethod
    def _check_can_cache(gm: torch.fx.GraphModule) -> None:
        """
        Check some conditions that would preclude caching and raise BypassFxGraphCache
        to bypass in case caching is not possible.
        """
        # Freezing can embed constants that wouldn't be static across runs.
        if config.freezing or config.aot_inductor.use_runtime_constant_folding:
            raise BypassFxGraphCache(
                "Freezing may introduce constants that aren't static across runs."
            )

        # The treatment of guards in the caching implementation requires that
        # we have a shape env.
        if FxGraphCache._get_shape_env() is None:
            log.debug("fx graph cache no shape env")
            raise BypassFxGraphCache("No shape env.")

        # HigherOrderOperators should be handled on a case-by-case basis.
        # Currently, we just skip caching if we have any.
        # We also skip if there are any torchbind objects.
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.HigherOrderOperator):
                raise BypassFxGraphCache("Can't cache HigherOrderOperators.")
            if node.op == "getattr" and isinstance(
                getattr(gm, node.target), torch._C.ScriptObject
            ):
                raise BypassFxGraphCache("Can't cache torchbind objects.")

    @staticmethod
    def load(  # type: ignore[no-untyped-def]
        compile_fx_fn: Callable[..., Any],
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
        inputs_to_check: Sequence[int],
        local: bool,
        remote: bool,
    ):
        """
        Load a compiled graph from the cache. If a cached entry does not exist,
        compile the graph and save it to the cache.
        """
        assert local or remote, "at least one of them needs to be enabled"
        compiled_graph = None
        cache_state = None
        cache_event_time = None
        cache_info: Dict[str, Any] = {}
        try:
            FxGraphCache._check_can_cache(gm)
            key, debug_lines = compiled_fx_graph_hash(
                gm, example_inputs, fx_kwargs, inputs_to_check
            )
            cache_info["key"] = key
            cache_info["components"] = debug_lines

            remote_cache: Optional[RemoteCache[JsonDataTy]] = None
            if remote:
                cache_id = "fx-graph-v1"
                try:
                    if config.is_fbcode():
                        from torch._inductor.fb.remote_cache import FbRemoteFxGraphCache

                        remote_cache = FbRemoteFxGraphCache(cache_id)
                    else:
                        from torch._inductor.remote_cache import RemoteFxGraphCache

                        remote_cache = RemoteFxGraphCache(cache_id)
                except ModuleNotFoundError as e:
                    # No need for a stack trace on this error
                    remote_cache = None
                    log.warning("Unable to create a remote cache: %s", e)
                except Exception:
                    remote_cache = None
                    log.warning("Unable to create a remote cache", exc_info=True)

            compiled_graph = FxGraphCache._lookup_graph(
                key, example_inputs, local, remote_cache
            )

            if compiled_graph is None:
                log.debug("fx graph cache miss for key %s", key)
                counters["inductor"]["fxgraph_cache_miss"] += 1
                cache_state = "miss"
                start_time = time_ns()
                cache_event_time = start_time
                compiled_graph = compile_fx_fn(
                    gm, example_inputs, inputs_to_check, fx_kwargs
                )
                compiled_graph._time_taken_ns = time_ns() - start_time
                cache_info["time_taken_ns"] = compiled_graph._time_taken_ns
                FxGraphCache._save_graph(
                    key,
                    compiled_graph,
                    example_inputs,
                    local,
                    remote_cache,
                )
            else:
                log.debug("fx graph cache hit for key %s", key)
                counters["inductor"]["fxgraph_cache_hit"] += 1
                cache_state = "hit"
                cache_event_time = time_ns()
                if (time_saved_ns := compiled_graph._time_taken_ns) is not None:
                    cache_info["time_saved_ns"] = time_saved_ns
                    if (
                        ephemeral_increase := add_ephemeral_timeout_increase_for_distributed(
                            time_saved_ns
                        )
                    ) != 0:
                        cache_info["ephemeral_timeout_increase"] = ephemeral_increase
            compiled_graph._fx_graph_cache_key = key
        except BypassFxGraphCache as e:
            counters["inductor"]["fxgraph_cache_bypass"] += 1
            cache_state = "bypass"
            log.info("Bypassing FX Graph Cache because '%s'", e)
            cache_info["cache_bypass_reason"] = str(e)
            if remote:
                log_cache_bypass("bypass_fx_graph", str(e))
            cache_event_time = time_ns()

        if not compiled_graph:
            compiled_graph = compile_fx_fn(
                gm, example_inputs, inputs_to_check, fx_kwargs
            )
        assert compiled_graph is not None
        cache_info["cache_state"] = cache_state
        chromium_log = get_chromium_event_logger()
        chromium_log.log_instant_event(
            f"fx_graph_cache_{cache_state}", cache_event_time, metadata=cache_info
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hash",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )
        # Use the passed in cudagraphs so that we mutate the BoxedBool correctly
        FxGraphCache.post_compile(
            compiled_graph, example_inputs, fx_kwargs["cudagraphs"]
        )
        return compiled_graph

    @staticmethod
    def clear() -> None:
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(FxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            pass
```

### 同一个 key 下为何是“候选集合”

本地路径为 `<cache>/fxgraph/<key[1:3]>/<key>/`。目录中的每个叶子文件都是一份序列化
`CompiledFxGraph`，文件名由保存内容 hash 得到。同一个 graph key 可以因不同 guards
拥有多个叶子；lookup 对文件名排序后逐个反序列化，而不是假定目录里只有一个版本。

`_filter_backed_symints` 只留下带 hint 的 SymInt。unbacked symbol 不参与候选 guard，
因为当前无法用具体值安全求值。lookup 从这些 backed symbols 得到 `hints`，先用纯整数
判断候选，不把试探性失败写入当前 ShapeEnv。

### local 在前，remote 在后，但都进入同一 guard 循环

内部 generator 的顺序是：

1. `local=True` 时遍历本地排序后的所有叶子；单个文件损坏只记 warning，继续下一个；
2. 本地候选全部产出以后，若存在 remote cache，执行一次 `get(key)`，base64 解码并
   反序列化一个候选；remote 构造或读取异常同样记 warning，让查找退化为其余候选或
   miss；
3. 外层循环对每个产出的 candidate 使用同一套 guard 判定。

如果本地候选已命中，外层立即 break，generator 不再恢复执行，因此不会请求 remote。
“本地优先”不是先把两边候选全部下载再排序。

### 两个动态候选的具体选择

假设 key 相同，本地目录按文件名顺序有：

```text
C0.guards_expr = "s0 % 16 == 0"     # 无尾部版本
C1.guards_expr = "s0 % 16 != 0"     # 带尾部版本
当前 backed hints = [19]
```

对 C0，`evaluate_guards_expression(...,[19])` 得到 False，循环继续；对 C1 得到 True，
选择 C1 并停止。若 hints=[32]，C0 先命中，C1 不再读取。候选的 `guards_expr` 为空时
直接视为命中，表示该版本没有额外符号条件。

若本地只有 C0、当前值为 19，且 remote 候选 R 的 guard 对 19 为真，则 lookup 在 C0
拒绝后继续 remote 并命中 R。若 R 也拒绝或读取失败，`graph` 保持 None，返回 miss；
真正的完整编译由外层 `load` 决定。

### 为什么试探使用 hints，而命中后还要再求值一次

候选循环使用整数 hints，目的是让失败候选不污染当前 ShapeEnv。选中并恢复 callable
后，源码用真实 `symints` 再求值同一个 expression，并断言为 True；这次求值会把该
compiled version 的 guards 安装进当前环境。前者回答“能不能选”，后者完成“选择后
运行时必须遵守什么”。artifact 与 guard 回装的中间步骤在 K03 展开。

### K02 迁移题

- local 第一个文件损坏、第二个 guard 为真：warning 后仍命中第二个，不转 remote。
- local 所有 guard 为假、remote 读取异常：lookup 返回 None，外层记一次 miss 并编译；
  remote 故障本身不构造一个假 hit。
- 候选无 guards_expr：它在出现位置立即命中；若希望更特殊的候选优先，不能依赖本文
  未承诺的文件语义顺序，需要保存/候选设计保证正确性。
- 当前没有 ShapeEnv：不会走到 `_lookup_graph` 的 assert；前面的 `_check_can_cache`
  抛出 bypass，随后完整编译但不参与这一层 cache。

K02 到此完成。K03 继续从选中的磁盘 `CompiledFxGraph` 恢复 wrapper 文件、
`module.call`、真实符号 guards、metrics/counters 和输出代码记录。

## K03：命中 metadata 后，还要把 wrapper 恢复成当前进程的 callable

序列化 candidate 保存的是 wrapper 的 `cache_key`、`source_code`、line map、constants
和 guards，不保存 C++/Triton/Python callable。lookup 选中 graph 后，用 wrapper
`cache_key` 调用 `get_path(...,"py")`，得到 PyCodeCache 应有的 `.py` artifact 路径。

### wrapper 文件丢失时从 graph entry 自愈

若 `.py` 文件不存在，代码先建立父目录，再从 graph 中保存的 source 重写。C++ prefix
路径有一个跨 cache 目录的修正：source 若包含相同 basename、却不是当前
`cpp_prefix_path()`，正则把旧 include 改成当前绝对路径，然后 `write_atomic` 落盘。

```text
磁盘 graph candidate
  cache_key/source/linemap/constants
             │
             ├─ wrapper.py 仍存在 ───────────────┐
             └─ wrapper.py 缺失 → source 原子重建 │
                                                  v
                           PyCodeCache.load_by_key_path(...).call
                                                  │
                                                  v
                              graph.current_callable
```

这一步只恢复由该 graph entry 指向的 wrapper artifact，不重新运行 Lowering/Scheduler。
加载模块顶层仍可能从其各自 code cache 加载 C++/Triton kernel，这是 FX hit 后仍需完成的
当前进程初始化。

若 `load_by_key_path` 抛 `OSError`，lookup 记录错误并返回 None。外层把它当 cache miss
完整重编译；不会返回一个 `current_callable=None` 的半成品。其他未被这里捕获的加载
异常仍按正常错误传播，不能泛化成所有 artifact 失败都会静默重编译。

### guard 第二次求值会改变当前 ShapeEnv

K02 已用整数 hints 选出 candidate。callable 恢复后，若 `guards_expr` 非空，lookup 再
用真实 SymInt 列表求值，并断言结果为 True。这一次 ShapeEnv 可以安装 guards。例如
候选要求 `s0%16!=0`，当前 s0 hint=19：

```text
候选阶段：expression + [19] -> True，不污染环境
命中阶段：expression + [s0] -> True，把 s0%16!=0 加入当前 guards
```

如果同一 expression 用真实 symbols 居然不成立，assert 暴露内部不一致，而不是继续
执行不受保护的 callable。

### 恢复编译时的观察量，而不是重新执行编译

cache entry 保存 `metrics_deltas` 和 `counter_deltas`。命中后将它们加回当前进程，使测试
和日志观察到与正常编译相应的统计增量；这不表示 Lowering/Codegen 重新运行。随后保存
当前 output code、记录 artifact path 与结构化 trace，并返回已经带 callable 的 graph。

### 跨进程 run2 的状态链

实验 15 的第二个进程没有 run1 的 Python module cache：

```text
相同 graph details -> 相同 f... key
本地 candidate 的 guards_expr 为空 -> 不需要动态条件，直接选中
candidate.current_callable 原为 None
-> 定位/加载 wrapper .py
-> 取得新进程中的 module.call
-> 恢复统计；本例没有动态 guard 需要回装
-> CompiledFxGraph 可调用
-> result_sum=272.0，fxgraph_cache_hit=1
```

所以 hit 省掉的是整段图编译，不是“直接执行 pickle 中的函数”。即使 wrapper 文件被
删除，只要 graph entry 中 source 完整，路径可以重建；若 PyCodeCache 随后仍无法加载，
本次由 hit 候选退化成 miss/recompile。

### K03 迁移题

- graph entry 存在但 wrapper `.py` 被删：先重写 artifact，再尝试加载；计数只有加载
  成功并返回 graph 时才能走 hit 分支。
- wrapper 文件存在但 `load_by_key_path` 触发 OSError：lookup 返回 None，外层编译并
  保存新 entry；不能调用磁盘对象中的 None。
- candidate guard 为空：跳过两次 guard 求值，但仍必须恢复 artifact/callable 和统计。
- graph source 引用旧目录下同名 cpp prefix：只修正 include path，不改变 graph key
  或重新 codegen wrapper。

K03 到此完成。K04 转到 miss/save：为何只清空复制品的 callable，怎样生成 pruned
guards，以及 local 多叶子与 remote 单值怎样保存同一序列化内容。

## K04：miss 保存的是可重建 artifact，不会破坏当前正在返回的 callable

`load` 在 lookup 返回 None 时记录 miss，计时调用 `compile_fx_fn`，得到当前进程可立即
运行的 `compiled_graph`。`_save_graph` 的第一步是浅复制：

```text
compiled_graph                    disk_compiled_graph = copy(compiled_graph)
current_callable = module.call    current_callable = None
当前调用继续返回它               只序列化这一份
```

因此保存不是在原对象上清空 callable。run1 编译结束后能立刻执行并得到 272，正符合这
条对象关系。

### guard expression 来自本次实际编译留下的 ShapeEnv

保存前重新筛出 backed SymInt，调用 `get_pruned_guards(symints)`，只保留与这些输入符号
相关的 guards，再由 `produce_guards_expression` 生成可跨进程保存的表达式。Tensor 的
静态 tensor metadata 与动态维的符号身份已进入 graph hash details；本次动态 hint
没有作为 SymInt key 内容，这里保存的 guard 正是用来选择不同 compiled version。

假设编译 s0=32 时留下 `s0%16==0` 与无关的 s1 条件，而输入列表只含 s0：pruned
结果只为当前候选保存 s0 条件。以后同 key、s0=19 的 lookup 会拒绝它，并可保存另一
份尾部候选。

### 一次 pickle 同时供 local 与 remote 使用

磁盘副本加入 guards_expr 后只 pickle 一次。失败时增加 `fxgraph_cache_pickle_error` 并
返回；当前编译结果仍可运行，只是这次没有可复用 entry。

序列化成功后：

- local 开启：建立 key 子目录，用“序列化 CompiledFxGraph 内容的 SHA-256”作为叶子
  文件名，原子写入。guards 或其他 metadata 不同会形成不同叶子。
- remote cache 存在：把同一 bytes 做 base64，连同本次编译耗时毫秒数执行
  `put(key,cache_data)`。当前 remote 接口按 graph key 保存/取得一个值，不像本地目录
  明确遍历多个叶子。

两处配置都启用且前面的步骤不失败时，可以保存同一份 bytes。它们共享 graph key 与
序列化内容；本文只承诺一次 remote lookup 至多返回一个 candidate，不推断远端服务
如何覆盖同 key 的旧值。

### 写失败与编译失败属于不同出口

local mkdir/write 与后面的 remote put 位于同一个 `try`。任一步异常都会记录 cache
write error，已经完成的 `compiled_graph` 仍由 `load` 返回；但两边不是完全独立：
local 分支先抛异常时，控制流直接进入 `except`，本次不会继续执行后面的 remote put。
反过来，remote put 失败不会回滚已经成功写入的 local leaf。`compile_fx_fn` 自身失败
不在 `_save_graph` 的容错范围内，不能被描述成“缓存写失败但本次仍成功”。

### K04 状态推演

下面换成一个独立的动态 shape 状态推演；它不是实验 15 的静态 run1/run2，也不声称
当前编译器对这两个尺寸实测后必然产生如下表达式。先约定同一个 key 已经依次编译出
两个 guard 版本，再观察缓存怎样保存和选择它们：

```text
第一次 s0=32:
  original callable=A
  disk copy callable=None, guard=s0%16==0
  local leaf hash H0

第二次 s0=19（H0 guard 拒绝后 miss）:
  original callable=B
  disk copy callable=None, guard=s0%16!=0
  local leaf hash H1
```

本地 key 目录包含 H0、H1；下次按 guard 选择。若 H1 写盘失败，当前 B 仍运行，但下一
进程只能看到 H0，对 19 再次 miss。若 remote put 失败而 local H1 已成功，local lookup
仍可命中；若 local 写先失败，同一个 `try` 不会继续 remote put，因此两个后端都没有
得到 H1，下一进程仍需重新编译。

K04 到此完成。K05 打开 `load` 的统一控制流、bypass 检查、post-compile 与
`CompiledFxGraph` boxed callable，把 run1/run2/cond 三条路径走到同一返回口。


## K05：hit、miss、bypass 最终都交出同一种可恢复 graph

### 完整源码：`CompiledFxGraph` 保存哪些可恢复内容

先看 run1 刚编译完成的一份对象。`graph.compile_to_fn()` 先得到
`current_callable=module.call`，随后 `codegen_and_compile` 用它构造
`CompiledFxGraph`。构造时保存 wrapper key/source/linemap、device、mutation、
constants、output strides、禁用 cudagraph 原因和 metrics/counter deltas；此时
`guards_expr=None`，`cudagraph_info=None`、`fx_kwargs={}`、`inputs_to_check=()`，
编译耗时也还没有写入。

`codegen_and_compile` 返回前再补入本次 `cudagraph_info`、真实 `fx_kwargs`、
`inputs_to_check` 和 boxed forward device index；cache miss 的外层计时结束后再写
`_time_taken_ns`。保存时浅复制当前 graph，把副本的 `current_callable` 清空，并只在
副本上写入从当前 ShapeEnv 产生的 `guards_expr`。因此 run1 原对象仍是
`current_callable=module.call, guards_expr=None`；磁盘副本则是
`current_callable=None`。实验 15 使用静态输入，没有 `s0%16` 这类动态条件，因而
不能把前一节动态案例的 guard 填到这份磁盘副本上。run2 命中后用 wrapper artifact
恢复新的 `module.call`。

```python
# 教学源码：CompiledFxGraph 完整产品定义；本注释仅标识教学副本。
@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    cache_linemap: Optional[List[Tuple[int, str]]]
    device_types: Set[str]
    device_idxs: Set[int]
    mutated_inputs: Set[str]
    mutated_input_idxs: Set[int]
    constants: Dict[str, torch.Tensor]
    torchbind_constants: Dict[str, torch._C.ScriptObject]
    output_strides: Optional[List[Optional[Tuple[_StrideExprStr, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]

    cudagraph_info: Optional[CudagraphCachedInfo]
    fx_kwargs: Dict[str, Any]
    inputs_to_check: Sequence[int]
    boxed_forward_device_index: Optional[BoxedDeviceIndex]

    _time_taken_ns: Optional[int] = None
    _boxed_call: Optional[bool] = None
    _fx_graph_cache_key: Optional[str] = None

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        output_strides: List[Optional[Tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
    ) -> None:
        self.current_callable = current_callable
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.cache_linemap = graph.cache_linemap
        # TODO - ordered set
        self.device_types = set(graph.device_types)
        self.device_idxs = set(graph.device_idxs)
        self.mutated_inputs = set(graph.mutated_inputs)
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        self.constants = graph.constants
        self.torchbind_constants = graph.torchbind_constants
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None
        self.cudagraph_info = None
        self.fx_kwargs = {}
        self.inputs_to_check = ()
        self.boxed_forward_device_index = None

    def __call__(self, inputs: List[Any]) -> Any:
        assert self.current_callable is not None
        return self.current_callable(inputs)
```

### cache 前置检查决定“参与缓存”还是“直接编译”

`_check_can_cache` 在形成 key 前拒绝四类当前无法安全复用的请求：freezing/runtime
constant folding、没有 ShapeEnv、图中含 HigherOrderOperator、通过 getattr 取得
torchbind ScriptObject。它们抛 `BypassFxGraphCache`，不是返回 lookup miss。

HigherOrderOperator 的绕过是当前实现的保守边界，而不是已经证明“它永远不能缓存”。
以 `torch.cond` 为例，一个节点除普通 tensor 参数外还携带 true/false 分支图；其他
HigherOrderOperator 还可能携带子图、Python callable 或专用状态。普通路径的统一合同
只验证外层 GraphModule details、输入 SymInt guards 和一个 wrapper artifact。当前源码
没有为每一种 HOO 分别证明这些额外结构怎样稳定进入 key、怎样设 guard、怎样在新进程
恢复，所以 `_check_can_cache` 看到任意 HOO 都先 bypass，并注明以后应逐类支持。这是
根据当前检查范围作出的保守处理，不表示分支图本身无法编译。

实验 15 还单独运行：

```python
pred = torch.tensor(True)
x = torch.tensor([2.0, 4.0])
result = torch.cond(pred, lambda v: v + 1, lambda v: v - 1, [x])
```

图中的 `torch.cond` 是 HigherOrderOperator，因此状态是 bypass=1、miss=0、hit=0；
`load` 捕获 `BypassFxGraphCache` 后仍调用 `compile_fx_fn`。因为 `pred=True`，执行 true
分支，把 `[2.0,4.0]` 的每一项加 1，所以结果是 `[3.0,5.0]`。

### `load` 的三条控制流

共同入口先要求 local/remote 至少开启一个，再做前置检查、key 和 debug components。
remote=True 时尝试构造 remote client；模块不存在或构造异常会令 client=None，local
仍可继续。随后调用 `_lookup_graph`：

| 路径 | 中间状态 | 编译/保存 | 计数 |
| --- | --- | --- | --- |
| hit | lookup 返回已恢复 callable 的 graph | 不编译；记录可能节省的时间 | hit+1 |
| miss | lookup 返回 None | 计时 compile，保存 disk copy | miss+1 |
| bypass | key/check/reducer 抛 bypass | try 之外补做 compile，不保存 | bypass+1 |

hit 与 miss 都把 `_fx_graph_cache_key` 写回 graph；bypass 未形成安全 key，不写该字段。
三条路径最后统一记录 cache state 的 chromium/structured event，然后调用
`post_compile` 并返回 graph。

### post-compile 为什么命中与新编译都必须执行

cache 保存的是编译 artifact，不保存当前 tracing context 和本次输入对象上的后处理。
因此 hit/miss/bypass 都要：

1. 将输出 strides 安装到当前 tracing context；
2. 若本次请求 cudagraph，沿缓存的禁用原因同步 BoxedBool，或执行 cudagraph
   post-compile；
3. cudagraph 被关闭时仍按 `inputs_to_check` 处理输入 realignment。

这些结果不写回 cache entry。它们依赖当前进程、当前输入和本次 BoxedBool，若直接返回
磁盘 graph 而跳过，就会丢失运行合同。

### `CompiledFxGraph` 的字段分三次形成

第一步是构造函数：保存 callable、wrapper key/source/linemap、device、mutation、
constants、输出 stride、cudagraph 禁用原因和 metrics/counters，并把尚未得到的
guard、cudagraph info、fx kwargs、inputs_to_check 初始化为空。第二步由
`codegen_and_compile` 和 cache miss 计时器补入本次编译 metadata 与 `_time_taken_ns`。
第三步只发生在保存副本上：清空 callable，并写入裁剪后的 `guards_expr`。所以不能把
“磁盘对象最后拥有哪些字段”误写成“构造函数一次性保存了所有字段”，也不能说保存
副本只修改 callable。

`__call__(inputs)` 只做两件事：断言 callable 已恢复，再把同一 boxed list 交给它。
它不会把列表展开为 `*args`。`FxGraphCache.load` 本身只保证 callable 非空；回到真实
`_compile_fx_inner` 后才设置 `_boxed_call=True`，再把对象交给 forward/backward
compiler plumbing，最终由 AOTAutograd 一侧按 boxed 输入协议消费。直接单独调用
`load` 不能据此假定 boxed flag 已设置。

### 三个实验进程走到同一出口

```text
run1: lookup None
  -> compile -> original callable 可用 -> save callable=None 的 copy
  -> post_compile -> boxed call -> sum 272

run2: lookup candidate
  -> guards_expr 为空，restore wrapper module.call -> metrics
  -> post_compile -> boxed call -> sum 272

cond: check 发现 HigherOrderOperator
  -> bypass -> compile（不 lookup/save）
  -> post_compile -> boxed call -> [3.0,5.0]
```

这三条路最终都必须得到 `current_callable is not None`。hit 不是跳过所有加载，bypass
也不是失败；差异只在能否安全复用和是否写入这一层 cache。

### K05 迁移题

- remote client 构造失败但 local candidate 合法：仍可 local hit；remote=True 不保证
  实际一定访问远端。
- hit graph 保存了 cudagraph 禁用原因，而本次 BoxedBool 为真：post_compile 复现禁用，
  不会因当前请求重新强开。
- disk candidate 反序列化成功但 callable 恢复 OSError：lookup 返回 None，路径从预期
  hit 改成 miss/compile/save，最终仍交出可调用 graph。
- cache check 因 HOO bypass：没有 key 候选与保存，但 compile/post-compile/boxed call
  均照常发生。

至此，生成源码、模块加载、FX Graph Cache 的候选选择和 `CompiledFxGraph` 的恢复已经连成一次完整路径。

本篇是 Codegen 第三个层次的最后一个源码叶子。接下来返回
[缓存、编译与加载父篇](../README.md)，再按正式阅读顺序回到
[Inductor 总览](../../../README.md)核对四个一级模块的入口、出口和案例交接；不再从
这里继续展开 PyCodeCache、远端服务或底层编译器缓存的内部实现。


