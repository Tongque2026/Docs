"""Compare joint/post pass observations for inference and autograd compilation."""

from pathlib import Path

import torch
from torch._inductor import config


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
active_case = ""


def targets(graph):
    return [
        str(node.target)
        for node in graph.nodes
        if node.op == "call_function"
    ]


def joint_hook(graph):
    events.append(f"{active_case}:joint targets={targets(graph)}")


def post_hook(graph):
    events.append(f"{active_case}:post targets={targets(graph)}")


def inference_fn(x):
    return torch.sin(x) * 2


def training_fn(x):
    return (torch.sin(x) * x).sum()


with config.patch(
    {
        "force_disable_caches": True,
        "joint_custom_pre_pass": joint_hook,
        "post_grad_custom_pre_pass": post_hook,
    }
):
    active_case = "inference"
    inference_x = torch.tensor([0.0, 1.0, 2.0])
    inference_result = torch.compile(
        inference_fn, backend="inductor", fullgraph=True
    )(inference_x)
    torch.testing.assert_close(inference_result, inference_fn(inference_x))

    torch._dynamo.reset()
    active_case = "training"
    training_x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    training_result = torch.compile(
        training_fn, backend="inductor", fullgraph=True
    )(training_x)
    training_result.backward()
    eager_x = training_x.detach().clone().requires_grad_(True)
    eager_result = training_fn(eager_x)
    eager_result.backward()
    torch.testing.assert_close(training_result.detach(), eager_result.detach())
    torch.testing.assert_close(training_x.grad, eager_x.grad)

events.append(f"inference_result={inference_result.tolist()}")
events.append(f"training_result={training_result.item()}")
events.append(f"training_grad={training_x.grad.tolist()}")
text = "\n".join(events) + "\n"
(OUT / "pass_training_inference.txt").write_text(text)
print(text)
