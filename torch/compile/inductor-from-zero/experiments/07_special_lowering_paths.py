"""Observe algorithm/extern, fallback, and control-flow lowering paths."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
results = []
case_name = ""
original_run_node = GraphLowering.run_node


def describe(value):
    if isinstance(value, (list, tuple)):
        return type(value).__name__ + "[" + ", ".join(describe(x) for x in value) + "]"
    parts = []
    current = value
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        part = type(current).__name__
        if isinstance(current, ir.Buffer) and current.name:
            part += f"(name={current.name})"
        parts.append(part)
        if isinstance(current, (ir.TensorBox, ir.StorageBox)):
            current = current.data
        elif isinstance(current, ir.MultiOutput):
            current = current.inputs[0]
        elif isinstance(current, ir.BaseView):
            current = current.data
        else:
            break
    return "->".join(parts)


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op == "call_function":
        events.append(
            f"{case_name}/{self.name or 'root'}: {node.target} => {describe(result)}; "
            f"operations={[type(op).__name__ for op in self.operations]}"
        )
    return result


def mm_case(a, b):
    return torch.mm(a, b)


def fallback_case(x):
    return torch.histc(x, bins=4, min=0, max=4)


def cond_case(pred, x):
    def true_fn(v):
        return v + 1

    def false_fn(v):
        return v - 1

    return torch.cond(pred, true_fn, false_fn, [x])


cases = [
    ("mm", mm_case, (torch.arange(16.0).reshape(4, 4), torch.eye(4))),
    ("fallback_histc", fallback_case, (torch.tensor([0.5, 1.5, 1.8, 3.2]),)),
    ("cond_true", cond_case, (torch.tensor(True), torch.tensor([2.0, 4.0]))),
    ("cond_false", cond_case, (torch.tensor(False), torch.tensor([2.0, 4.0]))),
]

with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    for name, fn, args in cases:
        case_name = name
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(*args)
        torch.testing.assert_close(actual, fn(*args))
        results.append(f"{name}: result={actual.tolist()}")

(OUT / "special_lowering_paths.txt").write_text(
    "\n".join(events) + "\nresults:\n" + "\n".join(results) + "\n"
)
print((OUT / "special_lowering_paths.txt").read_text())
