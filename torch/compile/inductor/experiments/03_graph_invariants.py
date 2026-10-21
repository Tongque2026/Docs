"""Show two FX invariants maintained around Inductor graph transforms."""

import operator
import importlib
from pathlib import Path
from unittest import mock

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.fx_utils import FakeTensorUpdater
from torch._inductor.pattern_matcher import _args, stable_topological_sort
from torch._inductor.virtualized import V
from torch.utils._python_dispatch import TorchDispatchMode


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
lines = []

# A graph can temporarily have a consumer before its input. lint detects it,
# and Inductor's stable sort restores dependency order while preserving the
# relative order of nodes where dependencies allow it.
graph = torch.fx.Graph()
x = graph.placeholder("x")
y = graph.placeholder("y")
add = graph.call_function(operator.add, (y, x))
add.append(x)  # Move x after its consumer and deliberately break topo order.
lines.append("broken_order=" + ",".join(n.name for n in graph.nodes))
try:
    graph.lint()
except RuntimeError as exc:
    lines.append("lint_before_sort=" + str(exc).splitlines()[0])
stable_topological_sort(graph)
graph.lint()
lines.append("fixed_order=" + ",".join(n.name for n in graph.nodes))


# Already-valid independent nodes retain their relative order.
stable_graph = torch.fx.Graph()
stable_b = stable_graph.placeholder("b")
stable_a = stable_graph.placeholder("a")
stable_sum = stable_graph.call_function(operator.add, (stable_a, stable_b))
stable_topological_sort(stable_graph)
stable_order = [n.name for n in stable_graph.nodes]
lines.append("stable_independent_order=" + ",".join(stable_order))
assert stable_order == ["b", "a", "add"]


# Build a deliberately reversed dependency chain without executing it.
deep_graph = torch.fx.Graph()
deep_output = deep_graph.output(None)
deep_mul = deep_graph.call_function(operator.mul, ())
deep_add = deep_graph.call_function(operator.add, ())
deep_x = deep_graph.placeholder("x")
deep_y = deep_graph.placeholder("y")
deep_add.args = (deep_x, deep_y)
deep_mul.args = (deep_add, 2)
deep_output.args = (deep_mul,)
lines.append("deep_broken_order=" + ",".join(n.name for n in deep_graph.nodes))
stable_topological_sort(deep_graph)
deep_graph.lint()
deep_order = [n.name for n in deep_graph.nodes]
lines.append("deep_fixed_order=" + ",".join(deep_order))
assert deep_order == ["x", "y", "add", "mul", "output"]


# Two consumers waiting on the same dependency wake in their original order.
waiter_graph = torch.fx.Graph()
waiter_first = waiter_graph.call_function(torch.neg, ())
waiter_second = waiter_graph.call_function(torch.relu, ())
waiter_x = waiter_graph.placeholder("x")
waiter_first.args = (waiter_x,)
waiter_second.args = (waiter_x,)
stable_topological_sort(waiter_graph)
waiter_order = [n.name for n in waiter_graph.nodes]
lines.append("same_dependency_waiters=" + ",".join(waiter_order))
assert waiter_order == ["x", "neg", "relu"]


# Waiting on the last argument is an optimization, not an assumption that all
# other inputs are ready: an adversarial order can wake and requeue a node.
recheck_graph = torch.fx.Graph()
recheck_add = recheck_graph.call_function(operator.add, ())
recheck_y = recheck_graph.placeholder("y")
recheck_x = recheck_graph.placeholder("x")
recheck_add.args = (recheck_x, recheck_y)
pattern_matcher_module = importlib.import_module("torch._inductor.pattern_matcher")
real_args = pattern_matcher_module._args
arg_checks = []


def recording_args(node):
    arg_checks.append(node.name)
    return real_args(node)


with mock.patch.object(pattern_matcher_module, "_args", recording_args):
    stable_topological_sort(recheck_graph)
recheck_order = [n.name for n in recheck_graph.nodes]
recheck_count = arg_checks.count(recheck_add.name)
lines.append(
    "last_arg_recheck=" + ",".join(recheck_order) + f" checks={recheck_count}"
)
assert recheck_order == ["y", "x", "add"]
assert recheck_count == 3


# _args recursively visits positional/keyword containers and preserves repeats.
arg_graph = torch.fx.Graph()
arg_x = arg_graph.placeholder("x")
arg_y = arg_graph.placeholder("y")


def nested_target(a, values, *, info):
    return a


arg_node = arg_graph.call_function(
    nested_target,
    (arg_x, [arg_y, arg_x]),
    {"info": {"again": arg_y}},
)
arg_names = [node.name for node in _args(arg_node)]
lines.append("nested_args=" + ",".join(arg_names))
assert arg_names == ["x", "y", "x", "y"]


# Missing/external dependencies and cycles leave waiting nonempty and assert.
external_graph = torch.fx.Graph()
external_node = external_graph.placeholder("external")
missing_graph = torch.fx.Graph()
missing_user = missing_graph.call_function(torch.neg, (external_node,))
try:
    stable_topological_sort(missing_graph)
except AssertionError:
    lines.append("external_dependency=AssertionError")
else:
    raise AssertionError("external dependency unexpectedly sorted")

cycle_graph = torch.fx.Graph()
cycle_a = cycle_graph.call_function(torch.neg, ())
cycle_b = cycle_graph.call_function(torch.relu, ())
cycle_a.args = (cycle_b,)
cycle_b.args = (cycle_a,)
try:
    stable_topological_sort(cycle_graph)
except AssertionError:
    lines.append("cycle=AssertionError")
else:
    raise AssertionError("cycle unexpectedly sorted")

empty_graph = torch.fx.Graph()
stable_topological_sort(empty_graph)
lines.append("empty_graph=ok")


def fn(a, b):
    return a + b


# Editing Graph nodes does not rewrite GraphModule.forward until recompile().
gm = torch.fx.symbolic_trace(fn)
a = torch.tensor([2.0])
b = torch.tensor([3.0])
call = next(n for n in gm.graph.nodes if n.op == "call_function")
call.target = operator.mul
lines.append(f"before_recompile={gm(a, b).item()}")
gm.recompile()
gm.graph.lint()
lines.append(f"after_recompile={gm(a, b).item()}")
lines.append("generated_code=" + " ".join(gm.code.split()))


class RecordOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def meta_chain(a, b, c, d):
    z = torch.ops.aten.add.Tensor(a, b)
    return torch.ops.aten.relu.default(z)


# A changed input shape propagates through the changed add and its relu user.
meta_gm = make_fx(meta_chain, tracing_mode="fake")(
    torch.randn(2, 3),
    torch.randn(2, 3),
    torch.randn(4, 3),
    torch.randn(4, 3),
)
meta_nodes = list(meta_gm.graph.nodes)
meta_add = next(n for n in meta_nodes if n.target == torch.ops.aten.add.Tensor)
meta_relu = next(n for n in meta_nodes if n.target == torch.ops.aten.relu.default)
meta_c, meta_d = [n for n in meta_nodes if n.op == "placeholder"][2:]
meta_updater = FakeTensorUpdater(meta_gm.graph)
meta_mode = meta_add.meta["val"].fake_mode
lines.append(f"meta_before_add={list(meta_add.meta['val'].shape)}")
lines.append(f"meta_before_relu={list(meta_relu.meta['val'].shape)}")
meta_add.args = (meta_c, meta_d)
shape_log = RecordOps()
with shape_log, V.set_fake_mode(meta_mode):
    meta_updater.incremental_update()
lines.append("shape_change_ops=" + ",".join(shape_log.ops))
lines.append(f"meta_after_add={list(meta_add.meta['val'].shape)}")
lines.append(f"meta_after_relu={list(meta_relu.meta['val'].shape)}")
shape_second_log = RecordOps()
with shape_second_log, V.set_fake_mode(meta_mode):
    meta_updater.incremental_update()
lines.append("shape_second_update_ops=" + ",".join(shape_second_log.ops))


def equal_meta_chain(a, b):
    z = torch.ops.aten.add.Tensor(a, b)
    return torch.ops.aten.relu.default(z)


# Changing add to sub changes value semantics but not tensor metadata. The
# updater recomputes sub, sees an equivalent metadata contract, and does not
# invoke the unchanged relu user.
equal_gm = make_fx(equal_meta_chain, tracing_mode="fake")(
    torch.randn(2, 3), torch.randn(2, 3)
)
equal_add = next(
    n for n in equal_gm.graph.nodes if n.target == torch.ops.aten.add.Tensor
)
equal_updater = FakeTensorUpdater(equal_gm.graph)
equal_mode = equal_add.meta["val"].fake_mode
equal_add.target = torch.ops.aten.sub.Tensor
equal_log = RecordOps()
with equal_log, V.set_fake_mode(equal_mode):
    equal_updater.incremental_update()
lines.append("equal_meta_ops=" + ",".join(equal_log.ops))
equal_second_log = RecordOps()
with equal_second_log, V.set_fake_mode(equal_mode):
    equal_updater.incremental_update()
lines.append("equal_meta_second_ops=" + ",".join(equal_second_log.ops))


# dtype is not one of the fields compared by the current implementation.
dtype_gm = make_fx(equal_meta_chain, tracing_mode="fake")(
    torch.randn(2, 3), torch.randn(2, 3)
)
dtype_add = next(n for n in dtype_gm.graph.nodes if n.target == torch.ops.aten.add.Tensor)
dtype_updater = FakeTensorUpdater(dtype_gm.graph)
dtype_mode = dtype_add.meta["val"].fake_mode
dtype_before = dtype_add.meta["val"].dtype
dtype_add.target = torch.ops.aten.eq.Tensor
dtype_log = RecordOps()
with dtype_log, V.set_fake_mode(dtype_mode):
    dtype_updater.incremental_update()
dtype_after = dtype_add.meta["val"].dtype
lines.append(
    f"dtype_change_ops={','.join(dtype_log.ops)} before={dtype_before} after={dtype_after}"
)


# Missing input meta and non-ATen Python targets are skipped and retried later.
missing_gm = make_fx(equal_meta_chain, tracing_mode="fake")(
    torch.randn(2, 3), torch.randn(2, 3)
)
missing_add = next(
    n for n in missing_gm.graph.nodes if n.target == torch.ops.aten.add.Tensor
)
missing_input = next(n for n in missing_gm.graph.nodes if n.op == "placeholder")
missing_updater = FakeTensorUpdater(missing_gm.graph)
missing_mode = missing_add.meta["val"].fake_mode
missing_add.target = torch.ops.aten.sub.Tensor
del missing_input.meta["val"]
missing_log = RecordOps()
with missing_log, V.set_fake_mode(missing_mode):
    missing_updater.incremental_update()
lines.append("missing_fake_ops=" + ",".join(missing_log.ops))

python_gm = make_fx(equal_meta_chain, tracing_mode="fake")(
    torch.randn(2, 3), torch.randn(2, 3)
)
python_add = next(
    n for n in python_gm.graph.nodes if n.target == torch.ops.aten.add.Tensor
)
python_updater = FakeTensorUpdater(python_gm.graph)
python_add.target = operator.add
python_updater.incremental_update()
lines.append("python_target_meta_unchanged=" + str(python_add.meta["val"].dtype))


def alias_chain(a):
    z = torch.ops.aten.view.default(a, [2, 3])
    return torch.ops.aten.relu.default(z)


# A view's old storage is shared with its input. Replacing it with clone changes
# alias identity even though shape/stride/device stay equal, so users propagate.
alias_gm = make_fx(alias_chain, tracing_mode="fake")(torch.randn(2, 3))
alias_input = next(n for n in alias_gm.graph.nodes if n.op == "placeholder")
alias_view = next(
    n for n in alias_gm.graph.nodes if n.target == torch.ops.aten.view.default
)
alias_updater = FakeTensorUpdater(alias_gm.graph)
alias_mode = alias_view.meta["val"].fake_mode
old_alias_shared = (
    alias_view.meta["val"].untyped_storage()._cdata
    == alias_input.meta["val"].untyped_storage()._cdata
)
alias_view.target = torch.ops.aten.clone.default
alias_view.args = (alias_input,)
alias_log = RecordOps()
with alias_log, V.set_fake_mode(alias_mode):
    alias_updater.incremental_update()
new_alias_shared = (
    alias_view.meta["val"].untyped_storage()._cdata
    == alias_input.meta["val"].untyped_storage()._cdata
)
lines.append(
    "alias_change_ops="
    + ",".join(alias_log.ops)
    + f" old_shared={old_alias_shared} new_shared={new_alias_shared}"
)


def dce_case(x, y):
    torch.sin(y)  # Pure and unused.
    x.copy_(y)  # Side effect and unused as a data result.
    return x


dce_gm = make_fx(dce_case)(torch.ones(3), torch.zeros(3))
lines.append(
    "dce_before="
    + ",".join(str(n.target) for n in dce_gm.graph.nodes if n.op == "call_function")
)
dce_gm.graph.eliminate_dead_code()
dce_gm.graph.lint()
dce_gm.recompile()
dce_targets = [n.target for n in dce_gm.graph.nodes if n.op == "call_function"]
lines.append("dce_after=" + ",".join(str(target) for target in dce_targets))
assert torch.ops.aten.sin.default not in dce_targets
assert torch.ops.aten.copy_.default in dce_targets

assert "before_recompile=5.0" in lines
assert "after_recompile=6.0" in lines
assert list(meta_add.meta["val"].shape) == [4, 3]
assert list(meta_relu.meta["val"].shape) == [4, 3]
assert shape_log.ops == ["aten.add.Tensor", "aten.relu.default"]
assert equal_log.ops == ["aten.sub.Tensor"]
assert shape_second_log.ops == []
assert equal_second_log.ops == ["aten.sub.Tensor"]
assert dtype_log.ops == ["aten.eq.Tensor"]
assert dtype_before == torch.float32 and dtype_after == torch.float32
assert missing_log.ops == []
assert old_alias_shared and not new_alias_shared
assert alias_log.ops == ["aten.clone.default", "aten.relu.default"]
(OUT / "graph_invariants.txt").write_text("\n".join(lines) + "\n")
print((OUT / "graph_invariants.txt").read_text())
