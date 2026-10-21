"""Observe Scheduler ordering, DCE, mutation versions, and last usage."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""

original_toposort = Scheduler.topological_sort_schedule
original_dce = Scheduler.dead_node_elimination
original_ancestors = Scheduler.compute_ancestors
original_last_usage = Scheduler.compute_last_usage


def names(nodes):
    return [node.get_name() for node in nodes]


def observed_toposort(self, nodes):
    result = original_toposort(self, nodes)
    reverse_probe = original_toposort(self, list(reversed(nodes)))
    events.append(
        f"[{case_name}] topological_sort input={names(nodes)} "
        f"output={names(result)} reverse_probe={names(reverse_probe)}"
    )
    return result


def observed_dce(self):
    before = names(self.nodes)
    original_dce(self)
    events.append(
        f"[{case_name}] dead_node_elimination before={before} after={names(self.nodes)} "
        f"removed_operations={list(V.graph.removed_operations)} "
        f"removed_buffers={list(V.graph.removed_buffers)}"
    )


def observed_ancestors(self):
    original_ancestors(self)
    state = {
        node.get_name(): {
            "ancestors": list(node.ancestors),
            "order": [node.min_order, node.max_order],
            "reads": [f"{type(dep).__name__}:{dep.name}" for dep in node.read_writes.reads],
            "outputs": [
                f"{output.get_name()}->mutates:{list(output.get_mutations())}"
                for output in node.get_outputs()
            ],
        }
        for node in self.nodes
    }
    events.append(
        f"[{case_name}] ancestors={state} "
        f"mutation_renames={self.mutation_renames} "
        f"mutation_real_name={self.mutation_real_name}"
    )


def observed_last_usage(self):
    original_last_usage(self)
    state = {node.get_name(): list(node.last_usage) for node in self.nodes}
    events.append(
        f"[{case_name}] last_usage={state} "
        f"mutation_real_name={self.mutation_real_name}"
    )


def three_mm(x, first_weight, left_weight, right_weight):
    shared = torch.mm(x, first_weight)
    left = torch.mm(shared, left_weight)
    right = torch.mm(shared, right_weight)
    return left, right


def mutation(x, y):
    old = x + 1
    x.copy_(y)
    return old, x


def double_mutation(x, y, z):
    before = x + 1
    x.copy_(y)
    middle = x * 2
    x.copy_(z)
    return before, middle, x


patches = (
    mock.patch.object(Scheduler, "topological_sort_schedule", observed_toposort),
    mock.patch.object(Scheduler, "dead_node_elimination", observed_dce),
    mock.patch.object(Scheduler, "compute_ancestors", observed_ancestors),
    mock.patch.object(Scheduler, "compute_last_usage", observed_last_usage),
)

with config.patch({"force_disable_caches": True}), patches[0], patches[1], patches[2], patches[3]:
    case_name = "three_mm"
    x = torch.arange(6.0).reshape(2, 3)
    first_weight = torch.arange(12.0).reshape(3, 4)
    left_weight = torch.eye(4)
    right_weight = torch.full((4, 4), 0.25)
    left, right = torch.compile(
        three_mm, backend="inductor", fullgraph=True
    )(x, first_weight, left_weight, right_weight)

    case_name = "input_mutation"
    mutable_x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([9.0, 9.0, 9.0])
    old, returned_x = torch.compile(
        mutation, backend="inductor", fullgraph=True
    )(mutable_x, y)

    case_name = "double_mutation"
    twice_x = torch.tensor([0.0, 1.0, 2.0])
    twice_y = torch.tensor([4.0, 5.0, 6.0])
    twice_z = torch.tensor([7.0, 8.0, 9.0])
    before, middle, twice_returned_x = torch.compile(
        double_mutation, backend="inductor", fullgraph=True
    )(twice_x, twice_y, twice_z)

torch.testing.assert_close(left, torch.mm(torch.mm(x, first_weight), left_weight))
torch.testing.assert_close(right, torch.mm(torch.mm(x, first_weight), right_weight))
torch.testing.assert_close(old, torch.tensor([1.0, 2.0, 3.0]))
torch.testing.assert_close(returned_x, y)
torch.testing.assert_close(before, torch.tensor([1.0, 2.0, 3.0]))
torch.testing.assert_close(middle, torch.tensor([8.0, 10.0, 12.0]))
torch.testing.assert_close(twice_returned_x, twice_z)


class FakeUser:
    def __init__(self, name, is_weak=False):
        self.name = name
        self.is_weak = is_weak

    def get_name(self):
        return self.name


class FakeBuffer:
    def __init__(self, name, users):
        self.name = name
        self.users = users

    def get_name(self):
        return self.name


class FakeNode:
    def __init__(self, name, output_name, users, side_effects=False):
        self.name = name
        self.outputs = [FakeBuffer(output_name, users)]
        self.side_effects = side_effects
        self.pruned = False

    def get_name(self):
        return self.name

    def get_outputs(self):
        return self.outputs

    def has_side_effects(self):
        return self.side_effects

    def prune_weak_deps(self):
        self.pruned = True


fake_scheduler = Scheduler.__new__(Scheduler)
fake_scheduler.nodes = [
    FakeNode("dead_parent", "tmp0", [FakeUser("dead_leaf")]),
    FakeNode("dead_leaf", "tmp1", []),
    FakeNode("effect", "effect_out", [], side_effects=True),
    FakeNode("live", "result", [FakeUser("OUTPUT")]),
]
fake_graph = SimpleNamespace(removed_buffers=set(), removed_operations=set())
with V.set_graph_handler(fake_graph):
    original_dce(fake_scheduler)
events.append(
    "[controlled_dce] "
    f"remaining={names(fake_scheduler.nodes)} "
    f"removed_operations={sorted(fake_graph.removed_operations)} "
    f"removed_buffers={sorted(fake_graph.removed_buffers)}"
)

events.extend(
    [
        f"three_mm_shapes={list(left.shape)},{list(right.shape)}",
        f"old={old.tolist()}",
        f"mutated_x={mutable_x.tolist()}",
        f"double_before={before.tolist()}",
        f"double_middle={middle.tolist()}",
        f"double_mutated_x={twice_x.tolist()}",
    ]
)
summary = "\n".join(events) + "\n"
(OUT / "scheduler_order_lifetime.txt").write_text(summary)
print(summary)
