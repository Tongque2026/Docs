"""Observe pass-stage graphs and exercise the real recursive entry wrappers."""

import importlib
from pathlib import Path
from unittest import mock

import torch
from torch import fx, nn
from torch._inductor import config


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
snapshots = {}


def save_graph(label):
    def hook(graph):
        snapshots[label] = graph.python_code("self").src

    return hook


def fn(x):
    old = torch.sin(x)
    x.add_(1)
    return old, x


def leaf(name):
    """Create a tiny GraphModule used only as a recursive-pass test fixture."""
    graph = fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    gm = fx.GraphModule(nn.Module(), graph)
    gm._teaching_name = name
    return gm


def cond_graph(name, true_graph, false_graph):
    root = nn.Module()
    root.add_module("true_graph", true_graph)
    root.add_module("false_graph", false_graph)
    graph = fx.Graph()
    pred = graph.placeholder("pred")
    value = graph.placeholder("value")
    true_node = graph.get_attr("true_graph")
    false_node = graph.get_attr("false_graph")
    graph.call_function(
        torch.ops.higher_order.cond,
        args=(pred, true_node, false_node, (value,)),
    )
    graph.output(value)
    gm = fx.GraphModule(root, graph)
    gm._teaching_name = name
    return gm


def mixed_root(while_cond, while_body, true_graph, false_graph):
    """Put while before cond so sorted node order has an observable consequence."""
    root = nn.Module()
    root.add_module("while_cond", while_cond)
    root.add_module("while_body", while_body)
    root.add_module("true_graph", true_graph)
    root.add_module("false_graph", false_graph)
    graph = fx.Graph()
    pred = graph.placeholder("pred")
    value = graph.placeholder("value")
    while_cond_node = graph.get_attr("while_cond")
    while_body_node = graph.get_attr("while_body")
    graph.call_function(
        torch.ops.higher_order.while_loop,
        args=(while_cond_node, while_body_node, (value,), ()),
    )
    true_node = graph.get_attr("true_graph")
    false_node = graph.get_attr("false_graph")
    graph.call_function(
        torch.ops.higher_order.cond,
        args=(pred, true_node, false_node, (value,)),
    )
    graph.output(value)
    gm = fx.GraphModule(root, graph)
    gm._teaching_name = "root"
    return gm


def capture_recursive_entries():
    """Patch only the stage bodies; call the product recursive wrappers unchanged."""
    compile_fx = importlib.import_module("torch._inductor.compile_fx")
    while_cond = cond_graph("while_cond", leaf("wc_true"), leaf("wc_false"))
    true_graph = cond_graph("true_graph", leaf("t_true"), leaf("t_false"))
    root = mixed_root(while_cond, leaf("while_body"), true_graph, leaf("false_graph"))

    direct_names = list(compile_fx._get_subgraph_names(root))
    assert direct_names == ["while_cond", "while_body", "true_graph", "false_graph"]

    pre_trace = []

    def fake_pre(gm, example_inputs):
        pre_trace.append((gm._teaching_name, example_inputs))
        replacement = fx.GraphModule(gm, gm.graph)
        replacement._teaching_name = gm._teaching_name
        replacement._replacement_of = gm._teaching_name
        return replacement

    old_child_ids = {
        name: id(getattr(root, name))
        for name in ("while_cond", "while_body", "true_graph", "false_graph")
    }
    sentinel_inputs = [object()]
    with mock.patch.object(compile_fx, "pre_grad_passes", fake_pre):
        pre_result = compile_fx._recursive_pre_grad_passes(root, sentinel_inputs)

    expected_order = [
        "wc_true",
        "wc_false",
        "while_cond",
        "while_body",
        "t_true",
        "t_false",
        "true_graph",
        "false_graph",
        "root",
    ]
    assert [name for name, _ in pre_trace] == expected_order
    assert pre_trace[-1][1] is sentinel_inputs
    assert all(inputs is None for _, inputs in pre_trace[:-1])
    assert pre_result is not root
    child_replaced = {
        name: id(getattr(root, name)) != old_child_ids[name]
        for name in old_child_ids
    }
    assert all(child_replaced.values())

    joint_trace = []
    joint_ids_before = {name: id(module) for name, module in pre_result.named_modules()}

    def fake_joint(gm):
        joint_trace.append(gm._teaching_name)
        gm._joint_seen = True

    with mock.patch.object(compile_fx, "joint_graph_passes", fake_joint):
        joint_result = compile_fx._recursive_joint_graph_passes(pre_result)
    joint_ids_after = {name: id(module) for name, module in pre_result.named_modules()}
    assert joint_result is None
    assert joint_trace == expected_order
    assert joint_ids_after == joint_ids_before

    post_trace = []

    def fake_post(gm, is_inference):
        post_trace.append((gm._teaching_name, is_inference))
        gm._post_seen = is_inference

    with mock.patch.object(compile_fx, "post_grad_passes", fake_post):
        post_result = compile_fx._recursive_post_grad_passes(
            pre_result, is_inference=True
        )
    assert post_result is None
    assert [name for name, _ in post_trace] == expected_order
    assert all(is_inference for _, is_inference in post_trace)

    return [
        f"direct_names={direct_names}",
        "pre_order=" + " -> ".join(name for name, _ in pre_trace),
        "pre_inputs="
        + str(["root-inputs" if inputs is sentinel_inputs else inputs for _, inputs in pre_trace]),
        f"pre_root_replaced={pre_result is not root}",
        f"pre_direct_children_replaced={child_replaced}",
        "joint_order=" + " -> ".join(joint_trace),
        f"joint_return={joint_result}",
        f"joint_module_ids_unchanged={joint_ids_after == joint_ids_before}",
        "post_order=" + " -> ".join(name for name, _ in post_trace),
        f"post_flags={[flag for _, flag in post_trace]}",
        f"post_return={post_result}",
    ]


x = torch.tensor([0.0, 1.0, 2.0])
with config.patch(
    {
        "force_disable_caches": True,
        "pre_grad_custom_pass": save_graph("pre"),
        "joint_custom_pre_pass": save_graph("joint"),
        "post_grad_custom_post_pass": save_graph("post"),
    }
):
    actual_old, actual_x = torch.compile(fn, backend="inductor", fullgraph=True)(x)

torch.testing.assert_close(actual_old, torch.sin(torch.tensor([0.0, 1.0, 2.0])))
torch.testing.assert_close(actual_x, torch.tensor([1.0, 2.0, 3.0]))

recursive_lines = capture_recursive_entries()

text = "\n".join(
    [
        *(
            f"===== {label.upper()} =====\n{snapshots.get(label, '<not captured>')}"
            for label in ("pre", "joint", "post")
        ),
        f"old={actual_old.tolist()}",
        f"caller_x={x.tolist()}",
        "===== RECURSIVE WRAPPERS =====",
        *recursive_lines,
    ]
) + "\n"
(OUT / "pass_stage_graphs.txt").write_text(text)
print(text)
