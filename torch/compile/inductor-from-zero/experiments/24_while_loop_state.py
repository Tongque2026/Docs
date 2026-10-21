"""Observe WhileLoop subgraphs, outputs, and zero-iteration reuse protection."""

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
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ir.TensorBox, ir.StorageBox)):
            current = current.data
        else:
            return current
    return current


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op == "call_function" and node.target == torch.ops.higher_order.while_loop:
        outputs = [unwrap(x) for x in result]
        assert all(isinstance(x, ir.MultiOutput) for x in outputs)
        events.append(
            f"root_operations={[type(op).__name__ for op in self.operations]}"
        )
        events.append(f"output_names={[x.get_name() for x in outputs]}")
        events.append(
            f"never_reuse={[x.get_name() in self.never_reuse_buffers for x in outputs]}"
        )
        while_node = outputs[0].inputs[0]
        events.append(
            f"subgraphs={while_node.cond_subgraph.name},{while_node.body_subgraph.name}"
        )
    return result


def loop_case(i, x):
    def cond_fn(iteration, value):
        return iteration < 3

    def body_fn(iteration, value):
        return iteration + 1, value + 2

    return torch.while_loop(cond_fn, body_fn, (i, x))


with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    compiled = torch.compile(loop_case, backend="inductor", fullgraph=True)
    three_iterations = compiled(torch.tensor(0), torch.tensor([1.0, 2.0]))
    zero_iterations = compiled(torch.tensor(3), torch.tensor([1.0, 2.0]))

torch.testing.assert_close(three_iterations[0], torch.tensor(3))
torch.testing.assert_close(three_iterations[1], torch.tensor([7.0, 8.0]))
torch.testing.assert_close(zero_iterations[0], torch.tensor(3))
torch.testing.assert_close(zero_iterations[1], torch.tensor([1.0, 2.0]))
events.extend(
    [
        f"three_iterations={[x.tolist() for x in three_iterations]}",
        f"zero_iterations={[x.tolist() for x in zero_iterations]}",
    ]
)

artifact = OUT / "while_loop_state.txt"
artifact.write_text("\n".join(events) + "\n")
print(artifact.read_text(), end="")
