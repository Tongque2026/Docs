"""Observe GraphLowering's per-node results for one current-source graph."""

import contextlib
import importlib
import operator
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor import ir
from torch._inductor.virtualized import V


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
events = []
original_run_node = GraphLowering.run_node


def type_chain(value):
    names = [type(value).__name__]
    while isinstance(value, (ir.TensorBox, ir.StorageBox)):
        value = value.data
        names.append(type(value).__name__)
    return "->".join(names)


def observed_run_node(self, node):
    result = original_run_node(self, node)
    if node.op == "output":
        result_desc = "None; graph_outputs=" + ",".join(
            type_chain(value) for value in self.graph_outputs
        )
    else:
        result_desc = type_chain(result)
    events.append(
        f"{node.op:13} {node.name:10} target={node.target} "
        f"result={result_desc} buffers={len(self.buffers)} "
        f"operations={len(self.operations)} env_inputs={list(self.graph_inputs)}"
    )
    return result


def fn(x, y):
    a = x + y
    b = torch.relu(a)
    return b * 2


x = torch.tensor([-2.0, 1.0, 3.0])
y = torch.tensor([1.0, 4.0, -1.0])
with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "run_node", observed_run_node
):
    actual = torch.compile(fn, backend="inductor", fullgraph=True)(x, y)

torch.testing.assert_close(actual, fn(x, y))


def scalar_graph(value):
    graph = torch.fx.Graph()
    node = graph.placeholder("scalar")
    node.meta["val"] = value
    graph.output((node,))
    return torch.fx.GraphModule({}, graph)


def lower_scalar(value, *, shape_env=None, fake_mode=None):
    graph = GraphLowering(
        scalar_graph(value), example_inputs=[value], shape_env=shape_env
    )
    fake_context = (
        V.set_fake_mode(fake_mode) if fake_mode is not None else contextlib.nullcontext()
    )
    error = None
    try:
        with fake_context, V.set_graph_handler(graph):
            graph.run(value)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    input_value = graph.graph_inputs.get("scalar", "<not-registered>")
    outputs = getattr(graph, "graph_outputs", [])
    events.append(
        "scalar_input "
        f"value_type={type(value).__name__} "
        f"input_names={graph.graph_input_names} "
        f"input_keys={list(graph.graph_inputs)} "
        f"graph_input_type={type(input_value).__name__} "
        f"graph_input={input_value} "
        f"output_type={type(outputs[0]).__name__ if outputs else '<none>'} "
        f"error={error}"
    )
    return graph


lower_scalar(3)
lower_scalar(2.5)
lower_scalar(None)
lower_scalar(BackwardState())

shape_env = ShapeEnv()
symbolic_size = shape_env.create_unbacked_symint()
symbolic_mode = FakeTensorMode(shape_env=shape_env)
lower_scalar(symbolic_size, shape_env=shape_env, fake_mode=symbolic_mode)
lower_scalar(True)
lower_scalar(object())


def tensor_graph(output_input=True):
    graph = torch.fx.Graph()
    node = graph.placeholder("x")
    graph.output((node,) if output_input else (None,))
    return torch.fx.GraphModule({}, graph)


def lower_tensor(value, *, name=None, output_input=True, shape_env=None, fake_mode=None):
    gm = tensor_graph(output_input)
    placeholder_node = next(n for n in gm.graph.nodes if n.op == "placeholder")
    placeholder_node.meta["val"] = value
    lowering = GraphLowering(
        gm,
        example_inputs=[value],
        name=name,
        shape_env=shape_env,
    )
    fake_context = (
        V.set_fake_mode(fake_mode) if fake_mode is not None else contextlib.nullcontext()
    )
    with fake_context, V.set_graph_handler(lowering):
        lowering.run(value)
    key = next(iter(lowering.graph_inputs))
    tensor = lowering.graph_inputs[key]
    layout = tensor.data.data.layout
    events.append(
        "tensor_input "
        f"name={name} output_input={output_input} "
        f"input_names={lowering.graph_input_names} keys={list(lowering.graph_inputs)} "
        f"sizes={list(layout.size)} strides={list(layout.stride)} "
        f"symbolic={value._has_symbolic_sizes_strides} "
        f"dtype={layout.dtype} device={layout.device} "
        f"original_type={type(lowering.graph_inputs_original[key]).__name__} "
        f"device_types={list(lowering.device_types)} "
        f"device_map={[(str(k), v.name) for k, v in lowering.device_node_mapping.items()]} "
        f"aligned={key in lowering.aligned_inputs}"
    )
    return lowering


lower_tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3))
lower_tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3).t())
lower_tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3), name="branch")
lower_tensor(
    torch.arange(6, dtype=torch.float32).reshape(2, 3), output_input=False
)

dynamic_env = ShapeEnv()
dynamic_mode = FakeTensorMode(shape_env=dynamic_env)
dynamic_size = dynamic_env.create_unbacked_symint()
with dynamic_mode:
    dynamic_tensor = torch.empty((dynamic_size, 3))
lower_tensor(
    dynamic_tensor,
    shape_env=dynamic_env,
    fake_mode=dynamic_mode,
)


def make_attr_module():
    root = nn.Module()
    root.register_buffer("scalar_const", torch.tensor(2.0))
    root.register_buffer("small_const", torch.tensor([1.0, 2.0, 3.0]))
    root.register_buffer("table_const", torch.arange(12.0).reshape(3, 4))
    subgraph = torch.fx.Graph()
    sub_x = subgraph.placeholder("x")
    subgraph.output(sub_x)
    root.add_module("branch", torch.fx.GraphModule(nn.Module(), subgraph))
    graph = torch.fx.Graph()
    attrs = [
        graph.get_attr("scalar_const"),
        graph.get_attr("small_const"),
        graph.get_attr("table_const"),
        graph.get_attr("branch"),
    ]
    graph.output(tuple(attrs))
    return torch.fx.GraphModule(root, graph)


attr_gm = make_attr_module()
attr_lowering = GraphLowering(attr_gm, example_inputs=[])
with V.set_graph_handler(attr_lowering):
    scalar_attr = attr_lowering.get_attr("scalar_const", (), {})
    small_attr = attr_lowering.get_attr("small_const", (), {})
    table_attr = attr_lowering.get_attr("table_const", (), {})
    branch_attr = attr_lowering.get_attr("branch", (), {})
events.append(
    "get_attr_types="
    + ",".join(
        type_chain(value)
        for value in (scalar_attr, small_attr, table_attr, branch_attr)
    )
)
events.append(
    "get_attr_constants="
    + str(list(attr_lowering.constants))
    + " reprs="
    + str(list(attr_lowering.constant_reprs))
)

forced_lowering = GraphLowering(attr_gm, example_inputs=[])
with config.patch({"always_keep_tensor_constants": True}), V.set_graph_handler(
    forced_lowering
):
    forced_scalar = forced_lowering.get_attr("scalar_const", (), {})
    forced_small = forced_lowering.get_attr("small_const", (), {})
events.append(
    "forced_constant_types="
    + ",".join(type_chain(value) for value in (forced_scalar, forced_small))
    + " constants="
    + str(list(forced_lowering.constants))
)

# Constant naming deduplicates identical storage positions, not equal values.
naming = GraphLowering(attr_gm, example_inputs=[])
base = torch.arange(5.0)
same_view = base[:3]
offset_view = base[1:4]
independent_equal = same_view.clone()
with V.set_graph_handler(naming):
    first_name = naming.allocate_non_dup_const_name("layer.weight", same_view)
    same_name = naming.allocate_non_dup_const_name("alias", same_view)
    offset_name = naming.allocate_non_dup_const_name("layer.weight", offset_view)
    equal_name = naming.allocate_non_dup_const_name("layer.weight", independent_equal)
    numeric_name = naming.allocate_non_dup_const_name("1.bad-name", torch.ones(1))
events.append(
    "constant_names="
    f"first={first_name} same={same_name} offset={offset_name} "
    f"equal={equal_name} numeric={numeric_name}"
)


def make_empty_lowering():
    graph = torch.fx.Graph()
    graph.output(())
    return GraphLowering(torch.fx.GraphModule({}, graph), example_inputs=[])


dispatch_lowering = make_empty_lowering()
events.append(
    "container_getitem="
    + str(dispatch_lowering.call_function(operator.getitem, ((10, 20), 1), {}))
)


def passthrough(x):
    return ("passthrough", x)


passthrough._inductor_lowering_function = True
events.append(
    "passthrough=" + str(dispatch_lowering.call_function(passthrough, (7,), {}))
)

graph_module = importlib.import_module("torch._inductor.graph")
unknown_no_decomp = torch.ops.aten.linalg_eig.default
unknown_with_decomp = torch.ops.aten.addcdiv.default
with config.patch({"implicit_fallbacks": False}):
    for label, target in (
        ("with_decomp", unknown_with_decomp),
        ("without_decomp", unknown_no_decomp),
    ):
        try:
            dispatch_lowering.call_function(target, (), {})
        except Exception as exc:
            events.append(f"missing_{label}={type(exc).__name__}")

fallback_calls = []


def fake_make_fallback(target):
    fallback_calls.append(str(target))
    graph_module.lowerings[target] = lambda *args, **kwargs: "fallback-result"


graph_module.lowerings.pop(unknown_no_decomp, None)
with config.patch({"implicit_fallbacks": True}), mock.patch.object(
    graph_module, "make_fallback", fake_make_fallback
):
    fallback_result = dispatch_lowering.call_function(unknown_no_decomp, (), {})
events.append(
    f"implicit_fallback={fallback_result} make_calls={fallback_calls}"
)
graph_module.lowerings.pop(unknown_no_decomp, None)


def fail_lowering(*args, **kwargs):
    raise ValueError("lowering boom")


graph_module.lowerings[unknown_no_decomp] = fail_lowering
try:
    dispatch_lowering.call_function(unknown_no_decomp, (), {})
except Exception as exc:
    events.append(
        f"lowering_error={type(exc).__name__} message={str(exc).splitlines()[0]}"
    )
finally:
    graph_module.lowerings.pop(unknown_no_decomp, None)

for label, method in (
    ("call_module", dispatch_lowering.call_module),
    ("call_method", dispatch_lowering.call_method),
):
    try:
        method("unused", (), {})
    except AssertionError:
        events.append(label + "=AssertionError")

(OUT / "graph_lowering_nodes.txt").write_text("\n".join(events) + "\n")
print((OUT / "graph_lowering_nodes.txt").read_text())
