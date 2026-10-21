"""Separate three reasons that can change a lazy value into a buffer."""

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


def describe(value):
    current = value
    parts = []
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        if isinstance(current, ir.ComputedBuffer):
            parts[-1] += f"({current.get_name()})"
            current = current.data
        elif isinstance(current, (ir.TensorBox, ir.StorageBox, ir.BaseView)):
            current = current.data
        else:
            break
    return "->".join(parts)


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op in ("call_function", "output"):
        if isinstance(result, (tuple, list)):
            result_text = " | ".join(describe(item) for item in result)
        else:
            result_text = describe(result)
        events.append(
            f"{case_name}:{node.op}:{node.target}: result={result_text}; "
            f"operations={[op.get_operation_name() for op in self.operations]}; "
            f"mutated={list(self.mutated_buffers)}"
        )
    return result


def chain(x):
    a = x + 1
    return a * 2


def mm_consumer(x, weight):
    a = x + 1
    return torch.mm(a, weight)


def old_read_across_copy(x, y):
    old = x + 1
    x.copy_(y)
    return old * 2


def old_reader_reordered_fn(x, y):
    old = x + 1
    x.copy_(y)
    return old * 2


cases = [
    ("chain", chain, (torch.arange(6.0).reshape(2, 3),)),
    (
        "mm_consumer",
        mm_consumer,
        (torch.arange(6.0).reshape(2, 3), torch.ones(3, 2)),
    ),
    (
        "old_read_across_copy",
        old_read_across_copy,
        (torch.tensor([0.0, 1.0, 2.0]), torch.tensor([9.0, 9.0, 9.0])),
    ),
    (
        "old_reader_reordered",
        old_reader_reordered_fn,
        (torch.tensor([0.0, 1.0, 2.0]), torch.tensor([9.0, 9.0, 9.0])),
    ),
]


def place_old_reader_after_copy(graph):
    if case_name != "old_reader_reordered":
        return
    copy_node = next(
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default
    )
    mul_node = next(
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor
    )
    copy_node.append(mul_node)


with config.patch(
    {
        "force_disable_caches": True,
        "post_grad_custom_post_pass": place_old_reader_after_copy,
    }
), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    for name, fn, args in cases:
        case_name = name
        eager_args = tuple(arg.clone() for arg in args)
        expected = fn(*eager_args)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(*args)
        torch.testing.assert_close(actual, expected)
        if name in ("old_read_across_copy", "old_reader_reordered"):
            torch.testing.assert_close(args[0], args[1])

(OUT / "lowering_materialization.txt").write_text("\n".join(events) + "\n")
print((OUT / "lowering_materialization.txt").read_text())
