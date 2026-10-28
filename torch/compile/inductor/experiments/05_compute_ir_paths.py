"""Capture Pointwise, Reduction, and PermuteView IR from current Inductor."""

from pathlib import Path
from unittest import mock

import sympy
import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""
original_run_node = GraphLowering.run_node
original_expand_create = ir.ExpandView.create
original_view_create = ir.View.create


def describe(value):
    parts = []
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, ir.ComputedBuffer):
            layout = current.get_layout()
            parts[-1] += f"(size={list(layout.size)},stride={list(layout.stride)})"
            current = current.data
        elif isinstance(current, ir.Reduction):
            parts[-1] += (
                f"(ranges={current.ranges},reduction_ranges={current.reduction_ranges},"
                f"type={current.reduction_type})"
            )
            break
        elif isinstance(current, ir.Pointwise):
            parts[-1] += f"(ranges={current.ranges})"
            break
        elif isinstance(current, ir.ReinterpretView):
            layout = current.get_layout()
            parts[-1] += f"(size={list(layout.size)},stride={list(layout.stride)},offset={layout.offset})"
            current = current.data
        elif isinstance(current, ir.BaseView):
            parts[-1] += f"(size={list(current.get_size())})"
            current = current.data
        elif isinstance(current, (ir.TensorBox, ir.StorageBox)):
            current = current.data
        else:
            break
    return "->".join(parts)


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op == "call_function":
        events.append(
            f"{case_name}: {node.target} => {describe(result)}; "
            f"buffers={len(self.buffers)},operations={len(self.operations)}"
        )
    elif node.op == "output":
        events.append(
            f"{case_name}: output => "
            + " | ".join(describe(v) for v in self.graph_outputs)
        )
    return result


def observed_expand_create(x, new_size):
    result = original_expand_create(x, new_size)
    events.append(
        f"{case_name}: ExpandView.create input_size={list(x.get_size())} "
        f"new_size={list(new_size)} => {describe(result)}"
    )
    return result


def observed_view_create(x, new_size):
    result = original_view_create(x, new_size)
    events.append(
        f"{case_name}: View.create input={describe(x)} "
        f"new_size={list(new_size)} => {describe(result)}"
    )
    return result


def pointwise(x, y):
    return x + y


def reduction(x):
    return x.sum(dim=1)


def reduction_keepdim(x):
    return x.sum(dim=1, keepdim=True)


def broadcast_pointwise(x, y):
    return x + y


def transpose_view(x):
    return x.permute(1, 0)


def reshape_contiguous(x):
    return x.reshape(3, 2)


def reshape_pointwise(x):
    return (x + 1).reshape(3, 2)


def reshape_empty(x):
    return x.reshape(0, 3)


cases = [
    ("pointwise", pointwise, (torch.arange(6.0).reshape(2, 3), torch.ones(2, 3))),
    (
        "broadcast",
        broadcast_pointwise,
        (torch.arange(6.0).reshape(2, 3), torch.tensor([[10.0, 20.0, 30.0]])),
    ),
    ("reduction", reduction, (torch.arange(256.0).reshape(2, 128),)),
    (
        "reduction_keepdim",
        reduction_keepdim,
        (torch.arange(256.0).reshape(2, 128),),
    ),
    ("transpose", transpose_view, (torch.arange(6.0).reshape(2, 3),)),
    ("reshape_contiguous", reshape_contiguous, (torch.arange(6.0).reshape(2, 3),)),
    ("reshape_pointwise", reshape_pointwise, (torch.arange(6.0).reshape(2, 3),)),
    ("reshape_empty", reshape_empty, (torch.empty(0, 3),)),
]

with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
), mock.patch.object(
    ir.ExpandView, "create", side_effect=observed_expand_create
), mock.patch.object(ir.View, "create", side_effect=observed_view_create):
    for name, fn, args in cases:
        case_name = name
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(*args)
        torch.testing.assert_close(actual, fn(*args))


def empty_graph_module():
    graph = torch.fx.Graph()
    graph.output(())
    return torch.fx.GraphModule({}, graph)


direct_graph = GraphLowering(empty_graph_module(), example_inputs=[])
with V.set_graph_handler(direct_graph):
    combine = ir.View.dynamic_reshape_indexer([6], [2, 3])
    split = ir.View.dynamic_reshape_indexer([2, 3], [6])
    fallback = ir.View.dynamic_reshape_indexer([2, 3], [3, 2])
    combine_result = tuple(map(sympy.simplify, combine([1, 2])))
    split_result = tuple(map(sympy.simplify, split([5])))
    fallback_result = tuple(map(sympy.simplify, fallback([2, 1])))

    # Exercise Reduction.create's degenerate exits without relying on a frontend
    # decomposition. The loader is not evaluated on the zero-size path.
    zero_sum = ir.Reduction.create(
        torch.device("cpu"),
        torch.float32,
        torch.float32,
        lambda index, reduction_index: None,
        [sympy.Integer(2)],
        [sympy.Integer(0)],
        "sum",
    )
    one_argmax = ir.Reduction.create(
        torch.device("cpu"),
        torch.int64,
        torch.float32,
        lambda index, reduction_index: None,
        [sympy.Integer(2)],
        [sympy.Integer(1)],
        "argmax",
    )
    one_sum = ir.Reduction.create(
        torch.device("cpu"),
        torch.float32,
        torch.float32,
        lambda index, reduction_index: None,
        [sympy.Integer(2)],
        [sympy.Integer(1)],
        "sum",
    )
    small_sum = ir.Reduction.create(
        torch.device("cpu"),
        torch.float32,
        torch.float32,
        lambda index, reduction_index: None,
        [sympy.Integer(2)],
        [sympy.Integer(3)],
        "sum",
    )
    scalar_small_sum = ir.Reduction.create(
        torch.device("cpu"),
        torch.float32,
        torch.float32,
        lambda index, reduction_index: None,
        [sympy.Integer(1)],
        [sympy.Integer(3)],
        "sum",
    )
    try:
        ir.Reduction.create(
            torch.device("cpu"),
            torch.float32,
            torch.float32,
            lambda index, reduction_index: None,
            [sympy.Integer(2)],
            [sympy.Integer(0)],
            "max",
        )
    except AssertionError as exc:
        zero_max_error = str(exc)
    else:
        raise AssertionError("zero-size max unexpectedly produced IR")

assert combine_result == (5,)
assert split_result == (1, 2)
assert fallback_result == (1, 2)
events.extend(
    [
        f"direct_reindex [6]<-[2,3] index=[1,2] => {combine_result}",
        f"direct_reindex [2,3]<-[6] index=[5] => {split_result}",
        f"direct_reindex fallback [2,3]<-[3,2] index=[2,1] => {fallback_result}",
        f"direct_reduction zero-size sum => {describe(zero_sum)}",
        f"direct_reduction size-one sum => {describe(one_sum)}",
        f"direct_reduction size-one argmax => {describe(one_argmax)}",
        f"direct_reduction small sum output_numel=2 => {describe(small_sum)}",
        f"direct_reduction small sum output_numel=1 => {describe(scalar_small_sum)}",
        f"direct_reduction zero-size max => AssertionError: {zero_max_error}",
    ]
)

(OUT / "compute_ir_paths.txt").write_text("\n".join(events) + "\n")
print((OUT / "compute_ir_paths.txt").read_text())
