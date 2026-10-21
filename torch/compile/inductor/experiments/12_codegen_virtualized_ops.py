"""Record LoopBody replay through the active codegen ops handler."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBodyBlock
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""
active_node = None
replay_number = 0

original_scheduler_codegen = SchedulerNode.codegen
original_block_call = LoopBodyBlock.__call__


def short(value, limit=180):
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


class RecordingOps:
    def __init__(self, inner, label):
        self.inner = inner
        self.label = label

    def __getattr__(self, name):
        target = getattr(self.inner, name)

        def call(*args, **kwargs):
            result = target(*args, **kwargs)
            events.append(
                f"[{case_name}] {self.label} ops.{name} "
                f"args={short(args)} kwargs={short(kwargs)} -> {short(result)}"
            )
            return result

        return call


def observed_scheduler_codegen(self, index_vars):
    global active_node
    prior = active_node
    active_node = self.get_name()
    events.append(
        f"[{case_name}] scheduler_node.codegen node={active_node} "
        f"index_vars={short(index_vars)}\n"
        f"[{case_name}] loop_body_graph node={active_node}:\n"
        f"{self._body.root_block.debug_str(active_node)}"
    )
    try:
        return original_scheduler_codegen(self, index_vars)
    finally:
        active_node = prior


def observed_block_call(self):
    global replay_number
    if active_node is None:
        return original_block_call(self)
    replay_number += 1
    label = f"replay={replay_number} node={active_node}"
    handler = RecordingOps(V.get_ops_handler(), label)
    with V.set_ops_handler(handler):
        return original_block_call(self)


def save_code(code):
    (OUT / f"virtualized_ops_{case_name}_output.py").write_text(code)


def pointwise(x, y):
    return torch.relu(x + y) * 2


def reduction(x):
    return (x + 1).sum(dim=1)


with config.patch({"force_disable_caches": True}), mock.patch.object(
    SchedulerNode, "codegen", observed_scheduler_codegen
), mock.patch.object(
    LoopBodyBlock, "__call__", observed_block_call
), mock.patch.object(
    GraphLowering, "save_output_code", save_code
):
    case_name = "pointwise"
    x = torch.tensor([-2.0, 1.0, 3.0])
    y = torch.tensor([1.0, 4.0, -1.0])
    pointwise_result = torch.compile(
        pointwise, backend="inductor", fullgraph=True
    )(x, y)

    case_name = "reduction"
    reduction_input = torch.arange(256.0).reshape(2, 128)
    reduction_result = torch.compile(
        reduction, backend="inductor", fullgraph=True
    )(reduction_input)

torch.testing.assert_close(pointwise_result, torch.tensor([0.0, 10.0, 4.0]))
torch.testing.assert_close(reduction_result, (reduction_input + 1).sum(dim=1))

events.extend(
    [
        f"pointwise_result={pointwise_result.tolist()}",
        f"reduction_result={reduction_result.tolist()}",
    ]
)
summary = "\n".join(events) + "\n"
(OUT / "codegen_virtualized_ops.txt").write_text(summary)
print(summary)
