"""Exercise MultiTemplateBuffer's lazy timing cache without a GPU."""

from pathlib import Path

from torch._inductor.ir import MultiTemplateBuffer


OUT = Path(__file__).with_name("artifacts")
OUT.mkdir(exist_ok=True)

calls = 0


def compute_timings():
    global calls
    calls += 1
    return {"extern_mm": 8.0, "triton_32": 5.0}


# Avoid constructing a graph-owned Buffer: these two methods only require the
# timing fields initialized below, so this isolates their current product logic.
buffer = object.__new__(MultiTemplateBuffer)
buffer._choice_timings_fn = compute_timings
buffer._choice_timings = None

before = buffer._choice_timings
first = buffer.get_min_choice()
after_first = dict(buffer.choice_timings)
second = buffer.get_min_choice()

lines = [
    f"before={before}",
    f"first={first}",
    f"after_first={after_first}",
    f"second={second}",
    f"timing_calls={calls}",
]
assert first == ("triton_32", 5.0)
assert second == first
assert calls == 1

artifact = OUT / "multi_template_state.txt"
artifact.write_text("\n".join(lines) + "\n")
print(artifact.read_text(), end="")
