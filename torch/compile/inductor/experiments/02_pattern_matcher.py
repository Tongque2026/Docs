"""Exercise the current PatternMatcherPass with success and rejected cases."""

import operator
from pathlib import Path

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    CallModule,
    CallFunction,
    GraphPatternEntry,
    KeywordArg,
    MULTIPLE,
    PatternMatcherPass,
    ReplacementPatternEntry,
    is_match,
    register_graph_pattern,
)


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
aten = torch.ops.aten
matcher = PatternMatcherPass(pass_name="tutorial_distribute_mul")


def replacement(a, b):
    return a * 2 + b * 2


replacement_graph = make_fx(replacement)(torch.ones(3), torch.ones(3))

pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.add.Tensor,
        KeywordArg("x"),
        KeywordArg("y"),
        _users=MULTIPLE,
    ),
    2,
)


def is_float32(match):
    return match.kwargs["x"].meta["val"].dtype == torch.float32


@register_graph_pattern(pattern, extra_check=is_float32, pass_dict=matcher)
def distribute_mul(match, x, y):
    match.replace_with_graph(replacement_graph, [x, y])


def original(x, y):
    return (x + y) * 2


def wrong_constant(x, y):
    return (x + y) * 3


def shared_add(x, y):
    a = x + y
    return a * 2, torch.relu(a)


def across_mutation(x, y):
    a = x + y
    x.copy_(y)
    return a * 2


def no_anchor(x, y):
    return torch.relu(x + y)


def repeated_same(x):
    return x + x


def repeated_different(x, y):
    return x + y


def run_case(fn, dtype):
    x = torch.tensor([1, 2, 3], dtype=dtype)
    y = torch.tensor([10, 20, 30], dtype=dtype)
    gm = make_fx(fn)(x.clone(), y.clone())
    before = gm.print_readable(print_output=False)
    counters.clear()
    count = matcher.apply(gm)
    matcher_counters = dict(counters["inductor"])
    gm.graph.lint()
    gm.recompile()
    after = gm.print_readable(print_output=False)
    actual_x = x.clone()
    expect_x = x.clone()
    actual = gm(actual_x, y.clone())
    expected = fn(expect_x, y.clone())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_x, expect_x)
    regions = [
        (node.name, node.meta.get("mutation_region_id")) for node in gm.graph.nodes
    ]
    return before, after, count, regions, matcher_counters


f32_before, f32_after, f32_count, f32_regions, f32_counters = run_case(
    original, torch.float32
)
i64_before, i64_after, i64_count, i64_regions, i64_counters = run_case(
    original, torch.int64
)
constant_before, constant_after, constant_count, constant_regions, _ = run_case(
    wrong_constant, torch.float32
)
shared_before, shared_after, shared_count, shared_regions, _ = run_case(
    shared_add, torch.float32
)
mutation_before, mutation_after, mutation_count, mutation_regions, _ = run_case(
    across_mutation, torch.float32
)
anchor_before, anchor_after, anchor_count, anchor_regions, _ = run_case(
    no_anchor, torch.float32
)

same_arg = KeywordArg("same")
same_input_pattern = CallFunction(aten.add.Tensor, same_arg, same_arg)
same_gm = make_fx(repeated_same)(torch.ones(3))
different_gm = make_fx(repeated_different)(torch.ones(3), torch.ones(3))
same_node = next(n for n in same_gm.graph.nodes if n.target == aten.add.Tensor)
different_node = next(
    n for n in different_gm.graph.nodes if n.target == aten.add.Tensor
)
same_match = same_input_pattern.match(same_node)
different_match = same_input_pattern.match(different_node)


def probe_apply_control_flow():
    """Exercise apply branches that the replacement example does not reach."""
    # The empty-registry fast path runs before input validation.
    empty_count = PatternMatcherPass().apply(object())

    try:
        matcher.apply(object())
    except RuntimeError as exc:
        invalid_error = str(exc)
    else:
        raise AssertionError("nonempty matcher must reject an invalid input")

    graph_input_gm = make_fx(original)(torch.ones(3), torch.ones(3))
    counters.clear()
    graph_input_count = matcher.apply(graph_input_gm.graph)
    graph_input_gm.graph.lint()

    # CPU output is skipped only under the current disable_cpp_codegen condition.
    fallback_gm = make_fx(original)(torch.ones(3), torch.ones(3))
    counters.clear()
    with config.patch({"disable_cpp_codegen": True}):
        cpu_fallback_count = matcher.apply(fallback_gm)

    # Two independent anchors reveal the reverse graph order used by apply().
    reverse_matcher = PatternMatcherPass(pass_name="tutorial_reverse_order")
    reverse_pattern = CallFunction(aten.neg.default, KeywordArg("x"))
    reverse_order = []

    def record_reverse(match, x):
        reverse_order.append(match.nodes[-1].name)

    GraphPatternEntry(reverse_pattern, lambda match: True, record_reverse).register(
        reverse_matcher.patterns
    )

    def two_neg(x, y):
        return torch.neg(x), torch.neg(y)

    reverse_gm = make_fx(two_neg)(torch.ones(3), torch.ones(3))
    counters.clear()
    reverse_count = reverse_matcher.apply(reverse_gm)
    reverse_counters = dict(counters["inductor"])

    # The first entry erases the anchor.  The second entry sees _erased and is not run.
    erase_matcher = PatternMatcherPass(pass_name="tutorial_erased_anchor")
    erase_events = []

    def erase_first(match, x):
        erase_events.append("first")
        matched_node = match.nodes[-1]
        matched_node.replace_all_uses_with(x)
        match.erase_nodes()

    def should_not_run(match, x):
        erase_events.append("second")

    GraphPatternEntry(reverse_pattern, lambda match: True, erase_first).register(
        erase_matcher.patterns
    )
    GraphPatternEntry(reverse_pattern, lambda match: True, should_not_run).register(
        erase_matcher.patterns
    )
    erase_gm = make_fx(torch.neg)(torch.ones(3))
    counters.clear()
    erase_count = erase_matcher.apply(erase_gm)
    erase_gm.graph.lint()

    # call_module scans all module calls, then filters by the extracted module class.
    class ModulePair(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.relu(x))

    module_matcher = PatternMatcherPass(pass_name="tutorial_call_module")
    module_events = []
    module_pattern = CallModule(nn.ReLU, KeywordArg("x"))

    def record_module(match, x):
        module_events.append(match.nodes[-1].target)

    GraphPatternEntry(module_pattern, lambda match: True, record_module).register(
        module_matcher.patterns
    )
    module_gm = torch.fx.symbolic_trace(ModulePair())
    counters.clear()
    module_count = module_matcher.apply(module_gm)

    assert empty_count == 0
    assert graph_input_count == 1
    assert cpu_fallback_count == 0
    assert reverse_count == 2
    assert reverse_order == ["neg_1", "neg"]
    assert reverse_counters["pattern_matcher_count"] == 2
    assert reverse_counters["pattern_matcher_nodes"] == 2
    assert erase_count == 1
    assert erase_events == ["first"]
    assert module_count == 1
    assert module_events == ["relu"]

    return {
        "empty_registry_invalid_input_count": empty_count,
        "invalid_input_error": invalid_error,
        "graph_input_count": graph_input_count,
        "cpu_fallback_count": cpu_fallback_count,
        "reverse_count": reverse_count,
        "reverse_order": reverse_order,
        "reverse_counters": reverse_counters,
        "erase_count": erase_count,
        "erase_events": erase_events,
        "call_module_count": module_count,
        "call_module_events": module_events,
    }


control_flow = probe_apply_control_flow()


class ProbeMatch:
    """Minimum current interface used by replace_with_graph for surgery probes."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.erase_calls = 0

    def output_nodes(self):
        return self.outputs

    def erase_nodes(self):
        self.erase_calls += 1


def make_multi_output_original():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(aten.neg.default, (x,))
    second = graph.call_function(aten.relu.default, (x,))
    graph.output((first, second))
    return torch.fx.GraphModule(nn.Module(), graph), x, first, second


def pack_pair(x):
    return x + 10, x + 20


def make_packed_original(include_direct_user=False):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    packed = graph.call_function(pack_pair, (x,))
    first = graph.call_function(operator.getitem, (packed, 0))
    second = graph.call_function(operator.getitem, (packed, 1))
    if include_direct_user:
        graph.output((first, second, packed))
    else:
        graph.output((first, second))
    return torch.fx.GraphModule(nn.Module(), graph), x, packed


def probe_replace_with_graph():
    # Two old outputs map positionally to two replacement outputs.  The earliest
    # old output is the insertion point for every new call_function node.
    multi_gm, multi_x, old_first, old_second = make_multi_output_original()
    old_first.meta.update({"recompute": "r1", "ac_graph_id": "g1", "old": "first"})
    old_second.meta["old"] = "second"

    def two_outputs(x):
        first = x + 1
        return first * 2, x + 3

    multi_replacement = make_fx(two_outputs)(torch.ones(3))
    multi_match = ProbeMatch([old_first, old_second])
    ReplacementPatternEntry.replace_with_graph(
        multi_match, multi_gm.graph, multi_replacement, [multi_x]
    )
    multi_gm.graph.lint()
    multi_gm.recompile()
    multi_result = multi_gm(torch.tensor([-1.0, 2.0]))
    expected_multi = two_outputs(torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(multi_result, expected_multi)
    multi_calls = [n for n in multi_gm.graph.nodes if n.op == "call_function"]
    tagged = [
        (n.name, n.meta.get("recompute"), n.meta.get("ac_graph_id"))
        for n in multi_calls
    ]
    assert all(n.meta.get("val") is not None for n in multi_calls)
    assert all("tensor_meta" in n.meta for n in multi_calls)
    assert tagged[0][1:] == ("r1", "g1")
    assert tagged[1][1:] == ("r1", "g1")
    assert tagged[2][1:] == (None, None)
    assert "recompute" not in multi_x.meta
    assert multi_match.erase_calls == 1

    # One packed old output is unpacked by getitem users; each user is replaced
    # with the corresponding member of the replacement list.
    packed_gm, packed_x, packed = make_packed_original()
    packed_match = ProbeMatch([packed])

    def unpacked_replacement(x):
        return x + 1, x + 2

    unpacked_graph = make_fx(unpacked_replacement)(torch.ones(3))
    ReplacementPatternEntry.replace_with_graph(
        packed_match, packed_gm.graph, unpacked_graph, [packed_x]
    )
    packed_gm.graph.lint()
    packed_gm.recompile()
    packed_result = packed_gm(torch.tensor([4.0]))
    torch.testing.assert_close(
        packed_result, unpacked_replacement(torch.tensor([4.0]))
    )
    packed_targets = [
        n.target for n in packed_gm.graph.nodes if n.op == "call_function"
    ]
    assert operator.getitem not in packed_targets
    assert pack_pair not in packed_targets
    assert packed_match.erase_calls == 1

    # A direct non-getitem user is an explicit unsupported packed-output shape.
    bad_gm, bad_x, bad_packed = make_packed_original(include_direct_user=True)
    try:
        ReplacementPatternEntry.replace_with_graph(
            ProbeMatch([bad_packed]), bad_gm.graph, unpacked_graph, [bad_x]
        )
    except AssertionError as exc:
        packed_failure = str(exc)
    else:
        raise AssertionError("packed output with a direct user must fail")

    # A replacement graph without trace metadata takes all metadata from the
    # corresponding old output in replace(), rather than in Replacer.run_node().
    meta_graph = torch.fx.Graph()
    meta_x = meta_graph.placeholder("x")
    meta_old = meta_graph.call_function(aten.neg.default, (meta_x,))
    meta_old.meta.update({"old_only": 17, "recompute": "r2"})
    meta_graph.output(meta_old)
    meta_gm = torch.fx.GraphModule(nn.Module(), meta_graph)
    replacement_no_meta = torch.fx.Graph()
    replacement_x = replacement_no_meta.placeholder("x")
    replacement_out = replacement_no_meta.call_function(
        aten.relu.default, (replacement_x,)
    )
    replacement_no_meta.output(replacement_out)
    replacement_no_meta_gm = torch.fx.GraphModule(nn.Module(), replacement_no_meta)
    ReplacementPatternEntry.replace_with_graph(
        ProbeMatch([meta_old]), meta_gm.graph, replacement_no_meta_gm, [meta_x]
    )
    meta_gm.graph.lint()
    new_meta_node = next(
        n for n in meta_gm.graph.nodes if n.op == "call_function"
    )
    assert new_meta_node.meta["old_only"] == 17
    assert new_meta_node.meta["recompute"] == "r2"

    # Optional outputs exercise None/None and old-node/None replacement paths.
    optional_graph = torch.fx.Graph()
    optional_x = optional_graph.placeholder("x")
    optional_old = optional_graph.call_function(aten.neg.default, (optional_x,))
    optional_graph.output((optional_old, None))
    optional_gm = torch.fx.GraphModule(nn.Module(), optional_graph)

    def optional_replacement(x):
        return x + 1, None

    optional_replacement_gm = make_fx(optional_replacement)(torch.ones(3))
    ReplacementPatternEntry.replace_with_graph(
        ProbeMatch([optional_old, None]),
        optional_gm.graph,
        optional_replacement_gm,
        [optional_x],
    )
    optional_gm.graph.lint()
    optional_gm.recompile()
    optional_result = optional_gm(torch.tensor([2.0]))
    assert optional_result[1] is None
    torch.testing.assert_close(optional_result[0], torch.tensor([3.0]))

    removed_gm, removed_x, removed_first, removed_second = make_multi_output_original()

    def remove_first_output(x):
        return None, x + 3

    remove_first_gm = make_fx(remove_first_output)(torch.ones(3))
    ReplacementPatternEntry.replace_with_graph(
        ProbeMatch([removed_first, removed_second]),
        removed_gm.graph,
        remove_first_gm,
        [removed_x],
    )
    removed_gm.graph.lint()
    removed_gm.recompile()
    removed_result = removed_gm(torch.tensor([2.0]))
    assert removed_result[0] is None
    torch.testing.assert_close(removed_result[1], torch.tensor([5.0]))

    # Despite the Union annotation, the current Interpreter constructor needs
    # named_modules(), so a raw replacement Graph fails in this implementation.
    raw_target_gm, raw_x, raw_old, _ = make_multi_output_original()
    try:
        ReplacementPatternEntry.replace_with_graph(
            ProbeMatch([raw_old]), raw_target_gm.graph, replacement_no_meta, [raw_x]
        )
    except AttributeError as exc:
        raw_graph_error = str(exc)
    else:
        raise AssertionError("raw replacement Graph unexpectedly succeeded")

    return {
        "multi_result": tuple(value.tolist() for value in multi_result),
        "multi_call_order": [n.name for n in multi_calls],
        "multi_tags": tagged,
        "multi_input_tagged": "recompute" in multi_x.meta,
        "multi_erase_calls": multi_match.erase_calls,
        "packed_result": tuple(value.tolist() for value in packed_result),
        "packed_targets": [str(target) for target in packed_targets],
        "packed_erase_calls": packed_match.erase_calls,
        "packed_failure": packed_failure,
        "old_meta_fallback": dict(new_meta_node.meta),
        "optional_result": (optional_result[0].tolist(), optional_result[1]),
        "removed_result": (removed_result[0], removed_result[1].tolist()),
        "raw_graph_error": raw_graph_error,
    }


replace_flow = probe_replace_with_graph()

assert f32_count == 1
assert i64_count == 0
assert constant_count == 0
assert shared_count == 1
assert mutation_count == 0
assert anchor_count == 0
assert is_match(same_match)
assert not is_match(different_match)
assert f32_counters["pattern_matcher_count"] == 1
assert f32_counters["pattern_matcher_nodes"] == 2
assert i64_counters.get("pattern_matcher_count", 0) == 0

(OUT / "pattern_matcher.txt").write_text(
    "FLOAT32 BEFORE\n"
    + f32_before
    + "\nFLOAT32 AFTER\n"
    + f32_after
    + f"\nfloat32 matches={f32_count}\n"
    + f"float32 counters={f32_counters}\n"
    + f"float32 regions={f32_regions}\n"
    + "\nINT64 BEFORE\n"
    + i64_before
    + "\nINT64 AFTER\n"
    + i64_after
    + f"\nint64 matches={i64_count}\n"
    + f"int64 counters={i64_counters}\n"
    + f"int64 regions={i64_regions}\n"
    + "\nWRONG CONSTANT BEFORE\n"
    + constant_before
    + "\nWRONG CONSTANT AFTER\n"
    + constant_after
    + f"\nwrong_constant matches={constant_count}\n"
    + f"wrong_constant regions={constant_regions}\n"
    + "\nSHARED ADD BEFORE\n"
    + shared_before
    + "\nSHARED ADD AFTER\n"
    + shared_after
    + f"\nshared_add matches={shared_count}\n"
    + f"shared_add regions={shared_regions}\n"
    + "\nACROSS MUTATION BEFORE\n"
    + mutation_before
    + "\nACROSS MUTATION AFTER\n"
    + mutation_after
    + f"\nacross_mutation matches={mutation_count}\n"
    + f"across_mutation regions={mutation_regions}\n"
    + "\nNO ANCHOR BEFORE\n"
    + anchor_before
    + "\nNO ANCHOR AFTER\n"
    + anchor_after
    + f"\nno_anchor matches={anchor_count}\n"
    + f"no_anchor regions={anchor_regions}\n"
    + "\nREPEATED PATTERN\n"
    + f"same_input matched={is_match(same_match)}\n"
    + f"different_inputs matched={is_match(different_match)} failure={different_match}\n"
    + "\nAPPLY CONTROL FLOW\n"
    + "\n".join(f"{key}={value}" for key, value in control_flow.items())
    + "\n"
    + "\nREPLACE WITH GRAPH\n"
    + "\n".join(f"{key}={value}" for key, value in replace_flow.items())
    + "\n"
)
print((OUT / "pattern_matcher.txt").read_text())
