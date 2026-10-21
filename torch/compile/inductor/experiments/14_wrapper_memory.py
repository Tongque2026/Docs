"""Capture wrapper allocation, reuse, free, and call order for a multi-kernel graph."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)
case_name = "reuse"


def pipeline(x, weight0, weight1):
    first = torch.mm(x, weight0)
    activated = torch.relu(first)
    second = torch.mm(activated, weight1)
    return torch.sin(second)


def pool_pipeline(x, weight0, weight1):
    first = torch.mm(x, weight0)
    activated = torch.relu(first)
    second = torch.mm(activated, weight1)
    return torch.sin(second)


def save_code(code):
    filename = (
        "wrapper_memory_output.py"
        if case_name == "reuse"
        else "wrapper_memory_pool_output.py"
    )
    (OUT / filename).write_text(code)


torch.manual_seed(0)
x = torch.randn(8, 8)
weight0 = torch.randn(8, 8)
weight1 = torch.randn(8, 8)

with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "save_output_code", save_code
):
    actual = torch.compile(pipeline, backend="inductor", fullgraph=True)(
        x, weight0, weight1
    )

case_name = "pool"
with config.patch(
    {"force_disable_caches": True, "memory_planning": True}
), mock.patch.object(GraphLowering, "save_output_code", save_code):
    pool_actual = torch.compile(
        pool_pipeline, backend="inductor", fullgraph=True
    )(x, weight0, weight1)

expected = pipeline(x, weight0, weight1)
torch.testing.assert_close(actual, expected)
torch.testing.assert_close(pool_actual, expected)

code = (OUT / "wrapper_memory_output.py").read_text()
interesting = []
inside_call = False
for line in code.splitlines():
    stripped = line.strip()
    if stripped.startswith("def call("):
        inside_call = True
    if inside_call and any(
        token in stripped
        for token in (
            "assert_size_stride",
            "empty_strided",
            "reinterpret_tensor",
            "extern_kernels.",
            "cpp_fused_",
            "del ",
            "return ",
        )
    ):
        interesting.append(stripped)
    if inside_call and stripped.startswith("def benchmark_compiled_module"):
        break

pool_interesting = []
inside_pool_call = False
for line in (OUT / "wrapper_memory_pool_output.py").read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("def call("):
        inside_pool_call = True
    if inside_pool_call and any(
        token in stripped
        for token in (
            "assert_size_stride",
            "pool",
            "# reuse",
            "extern_kernels.",
            "cpp_fused_",
            "del ",
            "return ",
        )
    ):
        pool_interesting.append(stripped)
    if inside_pool_call and stripped.startswith("def benchmark_compiled_module"):
        break

summary = "\n".join(
    [
        f"result_shape={tuple(actual.shape)}",
        f"matches_eager={torch.allclose(actual, expected)}",
        "wrapper_lines:",
        *interesting,
        "pool_wrapper_lines:",
        *pool_interesting,
        f"pool_matches_eager={torch.allclose(pool_actual, expected)}",
    ]
) + "\n"
(OUT / "wrapper_memory.txt").write_text(summary)
print(summary)
