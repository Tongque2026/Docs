"""Observe Scheduler backend creation, flush boundaries, and codegen dispatch."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, test_operators
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.scheduler import Scheduler


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""

original_create_backend = Scheduler.create_backend
original_enter_context = Scheduler.enter_context
original_scheduler_flush = Scheduler.flush
original_extern = Scheduler.codegen_extern_call
original_codegen_node = CppScheduling.codegen_node
original_codegen_template = CppScheduling.codegen_template
original_cpp_flush = CppScheduling.flush


def state(scheduler):
    return (
        f"available={list(scheduler.available_buffer_names)} "
        f"completed={list(scheduler.completed_operations)} "
        f"pending_free={list(scheduler.buffer_names_to_free)}"
    )


def observed_create_backend(self, device):
    backend = original_create_backend(self, device)
    events.append(
        f"[{case_name}] create_backend device={device} type={type(backend).__name__}"
    )
    return backend


def observed_enter_context(self, node):
    original_enter_context(self, node)
    events.append(
        f"[{case_name}] visit {node.get_name()} type={type(node).__name__} "
        f"device={node.get_device()} current_device={self.current_device} "
        f"last_usage={list(node.last_usage)} {state(self)}"
    )


def observed_scheduler_flush(self):
    events.append(f"[{case_name}] scheduler.flush before {state(self)}")
    original_scheduler_flush(self)
    events.append(f"[{case_name}] scheduler.flush after {state(self)}")


def observed_extern(self, node):
    events.append(
        f"[{case_name}] dispatch extern {node.get_name()} "
        f"ir={type(node.node).__name__} {state(self)}"
    )
    return original_extern(self, node)


def observed_codegen_node(self, node):
    events.append(
        f"[{case_name}] dispatch cpp.codegen_node {node.get_name()} "
        f"type={type(node).__name__} members={[n.get_name() for n in node.get_nodes()]} "
        f"{state(self.scheduler)}"
    )
    return original_codegen_node(self, node)


def observed_codegen_template(self, template_node, epilogue_nodes):
    events.append(
        f"[{case_name}] dispatch cpp.codegen_template {template_node.get_name()} "
        f"epilogues={[n.get_name() for n in epilogue_nodes]} {state(self.scheduler)}"
    )
    return original_codegen_template(self, template_node, epilogue_nodes)


def observed_cpp_flush(self):
    scheduled = [node.get_name() for node in self.kernel_group.scheduled_nodes]
    events.append(
        f"[{case_name}] cpp.flush scheduled_nodes={scheduled} "
        f"ready={self.ready_to_flush()}"
    )
    return original_cpp_flush(self)


def pointwise(x):
    producer = test_operators.realize(x + 1)
    return producer * 2


def mixed(x, matrix, weight):
    pointwise_result = test_operators.realize(x + 1)
    mm_result = torch.mm(matrix, weight)
    return pointwise_result, torch.sin(mm_result)


def fallback(x):
    return torch.histc(x, bins=4, min=0, max=4)


def template_mm(x, weight):
    return torch.relu(torch.mm(x, weight))


class FrozenTemplateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.eye(64))

    def forward(self, x):
        return torch.relu(torch.mm(x, self.weight))


patches = (
    mock.patch.object(Scheduler, "create_backend", observed_create_backend),
    mock.patch.object(Scheduler, "enter_context", observed_enter_context),
    mock.patch.object(Scheduler, "flush", observed_scheduler_flush),
    mock.patch.object(Scheduler, "codegen_extern_call", observed_extern),
    mock.patch.object(CppScheduling, "codegen_node", observed_codegen_node),
    mock.patch.object(CppScheduling, "codegen_template", observed_codegen_template),
    mock.patch.object(CppScheduling, "flush", observed_cpp_flush),
)

with config.patch({"force_disable_caches": True}), patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
    case_name = "pointwise"
    x = torch.arange(8.0)
    pointwise_result = torch.compile(
        pointwise, backend="inductor", fullgraph=True
    )(x)

    case_name = "mixed"
    matrix = torch.arange(6.0).reshape(2, 3)
    weight = torch.arange(12.0).reshape(3, 4)
    mixed_result = torch.compile(
        mixed, backend="inductor", fullgraph=True
    )(x, matrix, weight)

    case_name = "fallback"
    hist_input = torch.tensor([0.0, 1.0, 1.5, 4.0])
    fallback_result = torch.compile(
        fallback, backend="inductor", fullgraph=True
    )(hist_input)

    case_name = "template_mm"
    template_x = torch.arange(256.0).reshape(16, 16)
    template_weight = torch.eye(16)
    with config.patch(
        {"max_autotune": True, "max_autotune_gemm_backends": "CPP"}
    ):
        template_result = torch.compile(
            template_mm, backend="inductor", fullgraph=True
        )(template_x, template_weight)

    case_name = "frozen_template_mm"
    frozen_template_x = torch.arange(4096.0).reshape(64, 64)
    frozen_template_module = FrozenTemplateModule().eval()
    with torch.no_grad(), config.patch(
        {
            "freezing": True,
            "max_autotune": True,
            "max_autotune_gemm_backends": "CPP",
        }
    ):
        frozen_template_result = torch.compile(
            frozen_template_module, backend="inductor", fullgraph=True
        )(frozen_template_x)

torch.testing.assert_close(pointwise_result, (x + 1) * 2)
torch.testing.assert_close(mixed_result[0], x + 1)
torch.testing.assert_close(mixed_result[1], torch.sin(torch.mm(matrix, weight)))
torch.testing.assert_close(fallback_result, torch.histc(hist_input, bins=4, min=0, max=4))
torch.testing.assert_close(template_result, torch.relu(torch.mm(template_x, template_weight)))
torch.testing.assert_close(
    frozen_template_result,
    torch.relu(torch.mm(frozen_template_x, frozen_template_module.weight)),
)

events.extend(
    [
        f"pointwise_result={pointwise_result.tolist()}",
        f"mixed_shapes={[list(value.shape) for value in mixed_result]}",
        f"fallback_result={fallback_result.tolist()}",
        f"template_shape={list(template_result.shape)}",
        f"frozen_template_shape={list(frozen_template_result.shape)}",
    ]
)
summary = "\n".join(events) + "\n"
(OUT / "scheduler_backend_dispatch.txt").write_text(summary)
print(summary)
