"""Capture one current-source Inductor pipeline without modifying PyTorch."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)


def fn(x, y):
    a = x + y
    return torch.relu(a) * 2


def save_post_grad(graph):
    gm = graph.owning_module
    (OUT / "post_grad_graph.txt").write_text(
        gm.print_readable(print_output=False, include_stride=True, include_device=True)
    )


def save_output_code(code):
    (OUT / "output_code.py").write_text(code)


x = torch.tensor([-2.0, 1.0, 3.0])
y = torch.tensor([1.0, 4.0, -1.0])

with config.patch(
    {
        "force_disable_caches": True,
        "post_grad_custom_post_pass": save_post_grad,
    }
), mock.patch.object(GraphLowering, "save_output_code", save_output_code):
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    actual = compiled(x, y)

expected = fn(x, y)
torch.testing.assert_close(actual, expected)
(OUT / "result.txt").write_text(
    f"x={x.tolist()}\ny={y.tolist()}\nexpected={expected.tolist()}\nactual={actual.tolist()}\n"
)
print((OUT / "post_grad_graph.txt").read_text())
print((OUT / "result.txt").read_text())
