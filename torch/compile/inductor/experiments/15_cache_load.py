"""Demonstrate an FX graph cache miss then hit in isolated processes."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)


if "--bypass-worker" in sys.argv:
    import torch
    from torch._dynamo.utils import counters

    def cond_fn(pred, x):
        return torch.cond(pred, lambda v: v + 1, lambda v: v - 1, [x])

    pred = torch.tensor(True)
    x = torch.tensor([2.0, 4.0])
    result = torch.compile(cond_fn, backend="inductor", fullgraph=True)(pred, x)
    torch.testing.assert_close(result, torch.tensor([3.0, 5.0]))
    cache_counts = counters["inductor"]
    print(f"fxgraph_cache_miss={cache_counts['fxgraph_cache_miss']}")
    print(f"fxgraph_cache_hit={cache_counts['fxgraph_cache_hit']}")
    print(f"fxgraph_cache_bypass={cache_counts['fxgraph_cache_bypass']}")
    print(f"result={result.tolist()}")
    raise SystemExit(0)


if "--worker" in sys.argv:
    import torch
    from torch._dynamo.utils import counters

    def fn(x):
        return torch.relu(x + 1) * 2

    x = torch.arange(-3.0, 16.0)
    result = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    torch.testing.assert_close(result, fn(x))
    cache_counts = counters["inductor"]
    print(f"fxgraph_cache_miss={cache_counts['fxgraph_cache_miss']}")
    print(f"fxgraph_cache_hit={cache_counts['fxgraph_cache_hit']}")
    print(f"fxgraph_cache_bypass={cache_counts['fxgraph_cache_bypass']}")
    print(f"result_sum={result.sum().item()}")
    raise SystemExit(0)


script = str(Path(__file__).resolve())
with tempfile.TemporaryDirectory(prefix="inductor_q44_") as cache_dir:
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    env["PYTHONPATH"] = "/work/pytorch"
    runs = []
    for number in (1, 2):
        completed = subprocess.run(
            [sys.executable, script, "--worker"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        runs.append(f"run={number}\n{completed.stdout.strip()}")

    bypass = subprocess.run(
        [sys.executable, script, "--bypass-worker"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    runs.append(f"run=bypass_cond\n{bypass.stdout.strip()}")

summary = "\n".join(runs) + "\n"
(OUT / "cache_load.txt").write_text(summary)
print(summary)
