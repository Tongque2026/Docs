"""Capture Scheduler nodes and dependency construction before fusion."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, dependencies
from torch._inductor.dependencies import MemoryDep
from torch._inductor.loop_body import LoopBody, MemoryUsageType
from torch._inductor.scheduler import Scheduler


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""
original_compute_dependencies = Scheduler.compute_dependencies
original_extract_read_writes = dependencies.extract_read_writes


def dep_text(dep):
    fields = [f"name={dep.name}"]
    if isinstance(dep, MemoryDep):
        fields.append(f"index={dep.index}")
        fields.append(f"ranges={dep.ranges}")
    if hasattr(dep, "mutating_buf"):
        fields.append(f"mutating_buf={dep.mutating_buf}")
    return f"{type(dep).__name__}({', '.join(fields)})"


def node_text(node):
    outputs = []
    for output in node.get_outputs():
        outputs.append(
            f"{output.get_name()} aliases={list(output.get_aliases())} "
            f"mutations={list(output.get_mutations())}"
        )
    reads = [dep_text(dep) for dep in node.read_writes.reads]
    writes = [dep_text(dep) for dep in node.read_writes.writes]
    unmet = [dep_text(dep) for dep in node.unmet_dependencies]
    return (
        f"{node.get_name()}: scheduler_type={type(node).__name__}; "
        f"ir_type={type(node.node).__name__}; outputs={outputs}; "
        f"reads={reads}; writes={writes}; unmet={unmet}"
    )


def observed_compute_dependencies(self):
    events.append(f"\n[{case_name}] before compute_dependencies")
    events.extend(node_text(node) for node in self.nodes)
    original_compute_dependencies(self)
    events.append(f"[{case_name}] after compute_dependencies")
    events.extend(node_text(node) for node in self.nodes)
    events.append(f"mutation_renames={self.mutation_renames}")
    events.append(f"mutation_real_name={self.mutation_real_name}")
    for node in self.nodes:
        for output in node.get_outputs():
            users = [
                f"{user.get_name()}(weak={user.is_weak},inplace={user.can_inplace})"
                for user in output.users
            ]
            events.append(f"users[{output.get_name()}]={users}")


def observed_extract_read_writes(fn, *argsizes, **kwargs):
    if case_name == "ordinary_reduction" and isinstance(fn, LoopBody):
        events.append(f"loop_body_argsizes={argsizes}")
        for usage_type in MemoryUsageType:
            entries = [
                (
                    entry.buffer_name,
                    entry.index_name,
                    getattr(entry, "mode", None),
                )
                for entry in fn.memory_usage[usage_type]
            ]
            if entries:
                events.append(f"memory_usage[{usage_type.name}]={entries}")
        events.append(f"indexing_exprs={fn.indexing_exprs}")
    return original_extract_read_writes(fn, *argsizes, **kwargs)


def ordinary(x):
    reduced = (x + 1).sum(dim=1)
    return reduced * 2


def extern_then_pointwise(x, weight):
    return torch.sin(torch.mm(x, weight))


def mutation(x, y):
    old = x + 1
    x.copy_(y)
    return old, x


def read_after_mutation(x, y):
    x.copy_(y)
    return x * 2


with (
    config.patch({"force_disable_caches": True}),
    mock.patch.object(Scheduler, "compute_dependencies", observed_compute_dependencies),
    mock.patch.object(
        dependencies, "extract_read_writes", observed_extract_read_writes
    ),
):
    case_name = "ordinary_reduction"
    ordinary_result = torch.compile(
        ordinary, backend="inductor", fullgraph=True
    )(torch.arange(6.0).reshape(2, 3))

    case_name = "extern_then_pointwise"
    extern_result = torch.compile(
        extern_then_pointwise, backend="inductor", fullgraph=True
    )(
        torch.arange(6.0).reshape(2, 3),
        torch.arange(12.0).reshape(3, 4),
    )

    case_name = "input_mutation"
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([9.0, 9.0, 9.0])
    old, mutation_result = torch.compile(
        mutation, backend="inductor", fullgraph=True
    )(x, y)

    case_name = "read_after_input_mutation"
    x_after = torch.tensor([0.0, 1.0, 2.0])
    y_after = torch.tensor([4.0, 5.0, 6.0])
    post_mutation_result = torch.compile(
        read_after_mutation, backend="inductor", fullgraph=True
    )(x_after, y_after)

torch.testing.assert_close(ordinary_result, torch.tensor([12.0, 30.0]))
torch.testing.assert_close(old, torch.tensor([1.0, 2.0, 3.0]))
torch.testing.assert_close(mutation_result, y)
torch.testing.assert_close(post_mutation_result, y_after * 2)
torch.testing.assert_close(x_after, y_after)

events.extend(
    [
        f"ordinary_result={ordinary_result.tolist()}",
        f"extern_result_shape={list(extern_result.shape)}",
        f"old={old.tolist()}",
        f"mutated_x={x.tolist()}",
        f"post_mutation_result={post_mutation_result.tolist()}",
        f"post_mutated_x={x_after.tolist()}",
    ]
)
summary = "\n".join(events) + "\n"
(OUT / "scheduler_dependencies.txt").write_text(summary)
print(summary)
