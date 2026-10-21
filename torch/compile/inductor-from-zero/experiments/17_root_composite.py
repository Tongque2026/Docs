"""Capture one root-level case combining view, pointwise, reduction, extern, and two outputs."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
original_run_node = GraphLowering.run_node


def describe(value):
    if isinstance(value, (tuple, list)):
        return type(value).__name__ + "[" + ", ".join(describe(x) for x in value) + "]"
    parts = []
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        part = type(current).__name__
        if isinstance(current, ir.Buffer) and current.name:
            part += f"(name={current.name})"
        if hasattr(current, "get_size"):
            part += f"(size={list(current.get_size())})"
        parts.append(part)
        if isinstance(current, (ir.TensorBox, ir.StorageBox)):
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
            f"{node.target} => {describe(result)}; "
            f"operations={[type(op).__name__ for op in self.operations]}"
        )
    elif node.op == "output":
        events.append(
            "output => " + " | ".join(describe(x) for x in self.graph_outputs)
        )
    return result


def composite(x, weight):
    view = x.permute(1, 0)
    shifted = torch.relu(view + 1)
    rows = shifted.sum(dim=1)
    projected = torch.mm(rows.unsqueeze(0), weight)
    return rows, projected


x = torch.arange(384.0).reshape(128, 3)
weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
expected = composite(x, weight)

with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    actual = torch.compile(composite, backend="inductor", fullgraph=True)(x, weight)

torch.testing.assert_close(actual, expected)
events.append(f"rows={actual[0].tolist()}")
events.append(f"projected={actual[1].tolist()}")
(OUT / "root_composite.txt").write_text("\n".join(events) + "\n")
print((OUT / "root_composite.txt").read_text())
