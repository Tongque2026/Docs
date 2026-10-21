"""Observe GraphLowering.run_node state and propagate_mutation contracts."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)


def empty_graph_module():
    graph = torch.fx.Graph()
    graph.output(())
    return torch.fx.GraphModule({}, graph)


def type_chain(value):
    parts = []
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, (ir.TensorBox, ir.StorageBox)):
            current = current.data
        elif isinstance(current, ir.BaseView):
            current = current.data
        else:
            break
    return "->".join(parts)


# Direct interface probe: only a replaced, schema-writable argument propagates.
helper_graph = GraphLowering(empty_graph_module(), example_inputs=[])
fx_graph = torch.fx.Graph()
dst_node = fx_graph.placeholder("dst")
src_node = fx_graph.placeholder("src")
copy_node = fx_graph.call_function(
    torch.ops.aten.copy_.default, (dst_node, src_node), {"non_blocking": False}
)
fx_graph.output(copy_node)

old_dst, new_dst = object(), object()
old_src, new_src = object(), object()
propagated = []


def record_call(target, args, kwargs):
    propagated.append((target, args, kwargs))
    return "copy-result"


with mock.patch.object(helper_graph, "call_function", side_effect=record_call):
    helper_graph.propagate_mutation(
        copy_node,
        (old_dst, old_src),
        {"non_blocking": False},
        (new_dst, new_src),
        {"non_blocking": False},
    )

assert len(propagated) == 1
assert propagated[0][0] is torch.ops.aten.copy_.default
assert propagated[0][1] == (old_dst, new_dst)

same_dst_calls = []
with mock.patch.object(
    helper_graph,
    "call_function",
    side_effect=lambda *args: same_dst_calls.append(args),
):
    helper_graph.propagate_mutation(
        copy_node,
        (old_dst, old_src),
        {"non_blocking": False},
        (old_dst, new_src),
        {"non_blocking": False},
    )
assert same_dst_calls == []


# Full compile: relu has mm and output users, so run_node materializes it.
events = []
original_run_node = GraphLowering.run_node


def observed_run_node(self, node):
    buffer_watermark = len(self.buffers)
    operation_watermark = len(self.operations)
    result = original_run_node(self, node)
    new_buffers = self.buffers[buffer_watermark:]
    new_operations = self.operations[operation_watermark:]
    users = [(user.op, str(user.target)) for user in node.users]
    events.append(
        f"node={node.name} op={node.op} target={node.target} users={users} "
        f"result={type_chain(result)} "
        f"new_buffers={[b.get_name() for b in new_buffers]} "
        f"new_operations={[op.get_operation_name() for op in new_operations]}"
    )
    return result


def fn(x, weight):
    activation = torch.relu(x)
    product = torch.mm(activation, weight)
    return activation, product


x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    activation, product = torch.compile(fn, backend="inductor", fullgraph=True)(
        x, weight
    )

expected_activation = torch.relu(x)
expected_product = torch.mm(expected_activation, weight)
torch.testing.assert_close(activation, expected_activation)
torch.testing.assert_close(product, expected_product)

relu_event = next(event for event in events if "target=aten.relu.default" in event)
assert "output" in relu_event and "aten.mm.default" in relu_event
assert "ComputedBuffer" in relu_event

summary = (
    "propagate_replaced_writable="
    + f"target={propagated[0][0]},args_are_old_new_dst="
    + f"{propagated[0][1] == (old_dst, new_dst)}\n"
    + f"propagate_same_writable_calls={len(same_dst_calls)}\n"
    + "run_node_events:\n"
    + "\n".join(events)
    + f"\nactivation={activation.tolist()}\n"
    + f"product={product.tolist()}\n"
)
(OUT / "run_node_contracts.txt").write_text(summary)
print(summary)
