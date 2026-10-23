"""Capture current Scheduler fusion candidates, decisions, scores, and results."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config, test_operators
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.scheduler import Scheduler, WhyNoFuse


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
case_name = ""

original_get_possible = Scheduler.get_possible_fusions
original_fuse_once = Scheduler.fuse_nodes_once
original_last_usage = Scheduler.compute_last_usage
original_why = WhyNoFuse.__call__
original_cpp_fuse = CppScheduling.fuse


def node_text(node):
    members = [member.get_name() for member in node.get_nodes()]
    origins = []
    for member in node.get_nodes():
        if member.node is not None:
            origins.extend(
                str(origin.target) for origin in member.node.get_origins()
            )
    return (
        f"{node.get_name()}:{type(node).__name__}"
        f"(members={members},origins={origins},"
        f"group={getattr(node, 'group', None)},"
        f"buffers={sorted(node.get_buffer_names())},"
        f"used={sorted(node.used_buffer_names())},"
        f"reads={sorted(map(str, node.read_writes.reads))},"
        f"writes={sorted(map(str, node.read_writes.writes))},"
        f"unmet={sorted(map(str, node.unmet_dependencies))},"
        f"ancestors={sorted(node.ancestors)})"
    )


def observed_get_possible(self, nodes):
    result = original_get_possible(self, nodes)
    pairs = []
    for left, right in result:
        priority = self.get_backend(left.get_device()).get_fusion_pair_priority(
            left, right
        )
        pairs.append(
            f"({left.get_name()},{right.get_name()},"
            f"score={self.score_fusion(left, right)},priority={priority})"
        )
    events.append(
        f"[{case_name}] candidates nodes={[node_text(node) for node in nodes]} "
        f"ordered_pairs={pairs}"
    )
    return result


def observed_fuse_once(self, nodes):
    before = [node_text(node) for node in nodes]
    result = original_fuse_once(self, nodes)
    events.append(
        f"[{case_name}] fuse_once before={before} "
        f"after={[node_text(node) for node in result]}"
    )
    return result


def observed_last_usage(self):
    original_last_usage(self)
    events.append(
        f"[{case_name}] final_nodes={[node_text(node) for node in self.nodes]}"
    )


def observed_why(self, reason, *args):
    original_why(self, reason, *args)
    rendered = reason % args
    events.append(
        f"[{case_name}] reject {self.node1.get_name()}+"
        f"{self.node2.get_name()}: {rendered}"
    )


def observed_cpp_fuse(self, node1, node2):
    before = (node_text(node1), node_text(node2))
    result = original_cpp_fuse(self, node1, node2)
    events.append(
        f"[{case_name}] merge current_pair={before} result={node_text(result)}"
    )
    return result


def vertical(x):
    producer = test_operators.realize(x + 1)
    return producer * 2


def horizontal(x):
    left = test_operators.realize(x + 1)
    right = test_operators.realize(x * 2)
    return left, right


def branched(x):
    shared = test_operators.realize(x + 1)
    left = test_operators.realize(shared * 2)
    right = test_operators.realize(shared * 3)
    return left, right


def shifted_index(x):
    producer = test_operators.realize(x + 1)
    return torch.roll(producer, 1)


def extern_boundary(x, weight):
    return torch.sin(torch.mm(x, weight))


def limited_vertical(x):
    producer = test_operators.realize(x + 1)
    return producer * 2


patches = (
    mock.patch.object(Scheduler, "get_possible_fusions", observed_get_possible),
    mock.patch.object(Scheduler, "fuse_nodes_once", observed_fuse_once),
    mock.patch.object(Scheduler, "compute_last_usage", observed_last_usage),
    mock.patch.object(WhyNoFuse, "__call__", observed_why),
    mock.patch.object(CppScheduling, "fuse", observed_cpp_fuse),
)

with config.patch(
    {"force_disable_caches": True, "benchmark_fusion": False}
), patches[0], patches[1], patches[2], patches[3], patches[4]:
    x = torch.arange(8.0)

    case_name = "vertical"
    vertical_result = torch.compile(
        vertical, backend="inductor", fullgraph=True
    )(x)

    case_name = "horizontal"
    horizontal_result = torch.compile(
        horizontal, backend="inductor", fullgraph=True
    )(x)

    case_name = "branched"
    branched_result = torch.compile(
        branched, backend="inductor", fullgraph=True
    )(x)

    case_name = "shifted_index"
    shifted_result = torch.compile(
        shifted_index, backend="inductor", fullgraph=True
    )(x)

    case_name = "extern_boundary"
    matrix = torch.arange(6.0).reshape(2, 3)
    weight = torch.arange(12.0).reshape(3, 4)
    extern_result = torch.compile(
        extern_boundary, backend="inductor", fullgraph=True
    )(matrix, weight)

    case_name = "max_fusion_size_one"
    with config.patch({"max_fusion_size": 1}):
        limited_result = torch.compile(
            limited_vertical, backend="inductor", fullgraph=True
        )(x)

torch.testing.assert_close(vertical_result, (x + 1) * 2)
torch.testing.assert_close(horizontal_result[0], x + 1)
torch.testing.assert_close(horizontal_result[1], x * 2)
torch.testing.assert_close(branched_result[0], (x + 1) * 2)
torch.testing.assert_close(branched_result[1], (x + 1) * 3)
torch.testing.assert_close(shifted_result, torch.roll(x + 1, 1))
torch.testing.assert_close(extern_result, torch.sin(torch.mm(matrix, weight)))
torch.testing.assert_close(limited_result, (x + 1) * 2)

events.extend(
    [
        f"vertical_result={vertical_result.tolist()}",
        f"horizontal_shapes={[list(value.shape) for value in horizontal_result]}",
        f"branched_shapes={[list(value.shape) for value in branched_result]}",
        f"shifted_result={shifted_result.tolist()}",
        f"extern_shape={list(extern_result.shape)}",
        f"limited_result={limited_result.tolist()}",
    ]
)
summary = "\n".join(events) + "\n"
(OUT / "scheduler_fusion.txt").write_text(summary)
print(summary)
