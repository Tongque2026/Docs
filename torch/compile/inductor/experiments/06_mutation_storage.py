"""Observe old-value protection and input mutation in current Inductor."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
original_run_node = GraphLowering.run_node


def unwrap(value):
    parts = []
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, ir.ComputedBuffer):
            parts[-1] += f"(name={current.name},layout={type(current.layout).__name__})"
            current = current.data
        elif isinstance(current, (ir.TensorBox, ir.StorageBox)):
            current = current.data
        elif isinstance(current, ir.BaseView):
            current = current.data
        else:
            break
    return "->".join(parts)


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op == "call_function":
        events.append(
            f"{node.target}: result={unwrap(result)}; "
            f"buffers={[b.get_name() for b in self.buffers]}; "
            f"operations={[op.get_operation_name() for op in self.operations]}; "
            f"mutated_buffers={list(self.mutated_buffers)}"
        )
    elif node.op == "output":
        output_desc = " | ".join(unwrap(v) for v in self.graph_outputs)
        events.append("output:" + (f" {output_desc}" if output_desc else ""))
    return result


def save_post_grad(graph):
    gm = graph.owning_module
    (OUT / "mutation_post_grad_graph.txt").write_text(
        gm.print_readable(print_output=False, include_stride=True, include_device=True)
    )


def save_code(code):
    (OUT / "mutation_output_code.py").write_text(code)


def fn(x, y):
    old = x + 1
    x.copy_(y)
    return old, x


def mutate_without_returning_x(x, y):
    x.copy_(y)
    return y + 1


def mutate_and_return_twice(x):
    x.add_(10)
    return x, x


x = torch.tensor([0.0, 1.0, 2.0])
y = torch.tensor([9.0, 9.0, 9.0])
x_ptr_before = x.data_ptr()
with config.patch(
    {"force_disable_caches": True, "post_grad_custom_post_pass": save_post_grad}
), mock.patch.object(GraphLowering, "run_node", observed_run_node), mock.patch.object(
    GraphLowering, "save_output_code", save_code
):
    old, returned_x = torch.compile(fn, backend="inductor", fullgraph=True)(x, y)

torch.testing.assert_close(old, torch.tensor([1.0, 2.0, 3.0]))
torch.testing.assert_close(x, y)
assert returned_x.data_ptr() == x.data_ptr() == x_ptr_before

events.append("--- mutate_without_returning_x ---")
x_not_returned = torch.tensor([2.0, 4.0, 6.0])
y_not_returned = torch.tensor([7.0, 8.0, 9.0])
with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    other_output = torch.compile(
        mutate_without_returning_x, backend="inductor", fullgraph=True
    )(x_not_returned, y_not_returned)

torch.testing.assert_close(x_not_returned, y_not_returned)
torch.testing.assert_close(other_output, y_not_returned + 1)
assert other_output.data_ptr() != x_not_returned.data_ptr()

events.append("--- mutate_and_return_twice ---")
x_returned_twice = torch.tensor([1.0, 2.0])
with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    returned_first, returned_second = torch.compile(
        mutate_and_return_twice, backend="inductor", fullgraph=True
    )(x_returned_twice)

torch.testing.assert_close(x_returned_twice, torch.tensor([11.0, 12.0]))
assert returned_first.data_ptr() == x_returned_twice.data_ptr()
assert returned_second.data_ptr() == x_returned_twice.data_ptr()


def empty_graph_module():
    graph = torch.fx.Graph()
    graph.output(())
    return torch.fx.GraphModule({}, graph)


direct_graph = GraphLowering(empty_graph_module(), example_inputs=[])
with torch._inductor.virtualized.V.set_graph_handler(direct_graph):
    insignificant_input = ir.TensorBox.create(
        ir.InputBuffer(
            "stride_src",
            ir.FixedLayout(
                torch.device("cpu"), torch.float32, [2, 1, 3], [3, 3, 1]
            ),
        )
    )
    insignificant_result = direct_graph.try_match_insignificant_strides(
        insignificant_input, (3, 99, 1)
    )
    insignificant_storage, insignificant_layout = ir.as_storage_and_layout(
        insignificant_result
    )

    significant_input = ir.TensorBox.create(
        ir.InputBuffer(
            "significant_src",
            ir.FixedLayout(
                torch.device("cpu"), torch.float32, [2, 2, 3], [6, 3, 1]
            ),
        )
    )
    significant_result = direct_graph.try_match_insignificant_strides(
        significant_input, (6, 99, 1)
    )

assert isinstance(insignificant_result.data, ir.ReinterpretView)
assert insignificant_storage.get_name() == "stride_src"
assert list(insignificant_layout.size) == [2, 1, 3]
assert list(insignificant_layout.stride) == [3, 99, 1]
assert insignificant_layout.offset == 0
assert significant_result is significant_input


class LayoutDecisionProbe:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def decide_layout(self):
        self.log.append(self.name)


layout_decisions = []
direct_graph.buffers = [
    LayoutDecisionProbe("buf_first", layout_decisions),
    LayoutDecisionProbe("buf_second", layout_decisions),
]
direct_graph.finalize()
assert layout_decisions == ["buf_first", "buf_second"]

summary = (
    "events:\n"
    + "\n".join(events)
    + f"\nold={old.tolist()}\n"
    + f"caller_x_after={x.tolist()}\n"
    + f"returned_x={returned_x.tolist()}\n"
    + f"returned_aliases_caller={returned_x.data_ptr() == x.data_ptr()}\n"
    + f"not_returned_caller_x_after={x_not_returned.tolist()}\n"
    + f"not_returned_other_output={other_output.tolist()}\n"
    + f"not_returned_output_aliases_x={other_output.data_ptr() == x_not_returned.data_ptr()}\n"
    + f"returned_twice_caller_x={x_returned_twice.tolist()}\n"
    + f"returned_twice_first_aliases_x={returned_first.data_ptr() == x_returned_twice.data_ptr()}\n"
    + f"returned_twice_second_aliases_x={returned_second.data_ptr() == x_returned_twice.data_ptr()}\n"
    + "insignificant_stride="
    + f"type={type(insignificant_result.data).__name__},storage={insignificant_storage.get_name()},"
    + f"size={list(insignificant_layout.size)},stride={list(insignificant_layout.stride)},"
    + f"offset={insignificant_layout.offset}\n"
    + f"significant_stride_same_object={significant_result is significant_input}\n"
    + f"finalize_order={layout_decisions}\n"
)
(OUT / "mutation_storage.txt").write_text(summary)
print(summary)
