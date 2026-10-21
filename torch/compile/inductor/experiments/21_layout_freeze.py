"""Direct probes for current Buffer/FlexibleLayout freeze contracts."""

from pathlib import Path

import torch
from torch._inductor import ir
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)


def empty_graph_module():
    graph = torch.fx.Graph()
    graph.output(())
    return torch.fx.GraphModule({}, graph)


def make_buffer(name):
    return ir.Buffer(
        name,
        ir.FlexibleLayout(torch.device("cpu"), torch.float32, [2, 3]),
    )


graph = GraphLowering(empty_graph_module(), example_inputs=[])
with V.set_graph_handler(graph):
    default = make_buffer("default")
    default.freeze_layout()

    fill_row = make_buffer("fill_row")
    fill_row.freeze_layout_with_fill_order([1, 0])

    fill_column = make_buffer("fill_column")
    fill_column.freeze_layout_with_fill_order([0, 1])

    stride_order = make_buffer("stride_order")
    stride_order.freeze_layout_with_stride_order([1, 0])

    same_order = make_buffer("same_order")
    same_order.freeze_layout_with_same_order([100, 1])

    exact = make_buffer("exact")
    exact.freeze_layout_with_exact_strides([10, 2])

    second_freeze_error = None
    try:
        exact.freeze_layout_with_fill_order([1, 0])
    except Exception as exc:
        second_freeze_error = type(exc).__name__

    def computed(name, inner_fn):
        return ir.ComputedBuffer(
            name,
            ir.FlexibleLayout(torch.device("cpu"), torch.float32, [2, 3]),
            ir.Pointwise(torch.device("cpu"), torch.float32, inner_fn, [2, 3]),
        )

    row_read = computed("row", lambda index: ops.load("x", 3 * index[0] + index[1]))
    column_read = computed(
        "column", lambda index: ops.load("x", index[0] + 2 * index[1])
    )
    broadcast_read = computed("broadcast", lambda index: ops.load("x", index[1]))
    no_read = computed("constant", lambda index: ops.constant(1, torch.float32))
    computed_buffers = [row_read, column_read, broadcast_read, no_read]
    fill_orders = {buf.name: buf.get_fill_order() for buf in computed_buffers}
    for buf in computed_buffers:
        buf.decide_layout()
    decided_strides = {buf.name: list(buf.get_stride()) for buf in computed_buffers}

    def tensor_box(name, layout):
        pointwise = ir.Pointwise(
            torch.device("cpu"),
            torch.float32,
            lambda index: ops.constant(1, torch.float32),
            list(layout.size),
        )
        return ir.TensorBox.create(ir.ComputedBuffer(name, layout, pointwise))

    flexible_box = tensor_box(
        "flexible_box",
        ir.FlexibleLayout(torch.device("cpu"), torch.float32, [2, 3]),
    )
    flexible_result = ir.ExternKernel.require_stride_order(flexible_box, [0, 1])

    matching_box = tensor_box(
        "matching_box",
        ir.FixedLayout(torch.device("cpu"), torch.float32, [2, 3], [3, 1]),
    )
    matching_result = ir.ExternKernel.require_stride_order(matching_box, [1, 0])

    mismatching_box = tensor_box(
        "mismatching_box",
        ir.FixedLayout(torch.device("cpu"), torch.float32, [2, 3], [3, 1]),
    )
    mismatching_result = ir.ExternKernel.require_stride_order(
        mismatching_box, [0, 1]
    )

    insignificant_box = tensor_box(
        "insignificant_box",
        ir.FixedLayout(
            torch.device("cpu"), torch.float32, [2, 1, 3], [3, 3, 1]
        ),
    )
    insignificant_result = ir.ExternKernel.require_exact_strides(
        insignificant_box, [3, 99, 1]
    )

    zero_box = tensor_box(
        "zero_box",
        ir.FixedLayout(torch.device("cpu"), torch.float32, [0, 3], [3, 1]),
    )
    zero_result = ir.ExternKernel.require_exact_strides(zero_box, [99, 7])

observed = {
    "default": list(default.get_stride()),
    "fill_row": list(fill_row.get_stride()),
    "fill_column": list(fill_column.get_stride()),
    "stride_order": list(stride_order.get_stride()),
    "same_order": list(same_order.get_stride()),
    "exact": list(exact.get_stride()),
}
assert observed == {
    "default": [3, 1],
    "fill_row": [3, 1],
    "fill_column": [1, 2],
    "stride_order": [3, 1],
    "same_order": [3, 1],
    "exact": [10, 2],
}
assert second_freeze_error == "AssertionError"
assert fill_orders == {
    "row": [1, 0],
    "column": [0, 1],
    "broadcast": [1, 0],
    "constant": None,
}
assert decided_strides == {
    "row": [3, 1],
    "column": [1, 2],
    "broadcast": [3, 1],
    "constant": [3, 1],
}
assert flexible_result is flexible_box
assert list(flexible_result.get_stride()) == [1, 2]
assert matching_result is matching_box
assert mismatching_result is not mismatching_box
assert list(mismatching_result.get_stride()) == [1, 2]
assert insignificant_result is insignificant_box
assert zero_result is zero_box

summary = "\n".join(f"{name}={stride}" for name, stride in observed.items())
summary += f"\nsecond_directed_freeze_error={second_freeze_error}\n"
summary += f"fill_orders={fill_orders}\n"
summary += f"decided_strides={decided_strides}\n"
summary += (
    "require_strides="
    f"flexible_same={flexible_result is flexible_box},"
    f"flexible_stride={list(flexible_result.get_stride())},"
    f"fixed_match_same={matching_result is matching_box},"
    f"fixed_mismatch_same={mismatching_result is mismatching_box},"
    f"fixed_mismatch_stride={list(mismatching_result.get_stride())},"
    f"size1_exact_same={insignificant_result is insignificant_box},"
    f"zero_same={zero_result is zero_box}\n"
)
(OUT / "layout_freeze.txt").write_text(summary)
print(summary)
