"""Compare an actual CPU kernel with Triton's source-defined block indexing."""

from pathlib import Path
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)


def pointwise(x):
    return torch.relu(x + 1) * 2


def save_code(code):
    (OUT / "backend_models_cpu_output.py").write_text(code)


x = torch.arange(-3.0, 16.0)
with config.patch({"force_disable_caches": True}), mock.patch.object(
    GraphLowering, "save_output_code", save_code
):
    actual = torch.compile(pointwise, backend="inductor", fullgraph=True)(x)

expected = pointwise(x)
torch.testing.assert_close(actual, expected)

# This is not a Triton execution.  It independently evaluates the indexing
# formula emitted by current TritonKernel code and used by current tests:
# xoffset = program_id * XBLOCK; xindex = xoffset + arange(XBLOCK);
# xmask = xindex < xnumel.
xnumel = 19
xblock = 8
rows = []
covered = []
for program_id in range((xnumel + xblock - 1) // xblock):
    indices = [program_id * xblock + lane for lane in range(xblock)]
    mask = [index < xnumel for index in indices]
    active = [index for index, keep in zip(indices, mask) if keep]
    covered.extend(active)
    rows.append(
        f"pid={program_id} indices={indices} "
        f"mask={[int(value) for value in mask]} active={active}"
    )

assert covered == list(range(xnumel))

summary = "\n".join(
    [
        f"cpu_input={x.tolist()}",
        f"cpu_result={actual.tolist()}",
        f"cpu_matches_eager={torch.equal(actual, expected)}",
        "triton_formula=xoffset=pid*XBLOCK; xindex=xoffset+arange(0,XBLOCK); xmask=xindex<xnumel",
        f"mapping_xnumel={xnumel} XBLOCK={xblock}",
        *rows,
        f"covered={covered}",
        "triton_execution=not_run_no_cuda_or_triton",
    ]
) + "\n"
(OUT / "backend_models.txt").write_text(summary)
print(summary)
