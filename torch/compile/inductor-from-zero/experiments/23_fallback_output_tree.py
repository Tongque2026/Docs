"""Observe single-Tensor and tuple fallback output trees."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""
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
    if node.op == "call_function" and node.target in (
        torch.ops.aten.histc.default,
        torch.ops.aten.sort.default,
    ):
        leaves = result if isinstance(result, (list, tuple)) else [result]
        descriptions = []
        for leaf in leaves:
            inner = unwrap(leaf)
            assert isinstance(inner, ir.MultiOutput)
            descriptions.append(
                f"{inner.get_name()}:indices={inner.indices}:input={inner.inputs[0].get_name()}"
            )
        events.append(
            f"{case_name}:operations={[type(op).__name__ for op in self.operations]};"
            f"outputs={descriptions}"
        )
    return result


def histc_case(x):
    return torch.histc(x, bins=4, min=0, max=4)


def sort_case(x):
    return torch.sort(x)


with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    case_name = "histc"
    histc = torch.compile(histc_case, backend="inductor", fullgraph=True)(
        torch.tensor([0.5, 1.5, 1.8, 3.2])
    )
    case_name = "sort"
    values, indices = torch.compile(sort_case, backend="inductor", fullgraph=True)(
        torch.tensor([3.0, 1.0, 2.0])
    )

torch.testing.assert_close(histc, torch.tensor([1.0, 2.0, 0.0, 1.0]))
torch.testing.assert_close(values, torch.tensor([1.0, 2.0, 3.0]))
torch.testing.assert_close(indices, torch.tensor([1, 2, 0]))
events.extend(
    [
        f"histc_result={histc.tolist()}",
        f"sort_values={values.tolist()}",
        f"sort_indices={indices.tolist()}",
    ]
)

artifact = OUT / "fallback_output_tree.txt"
artifact.write_text("\n".join(events) + "\n")
print(artifact.read_text(), end="")
