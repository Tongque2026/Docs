from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


async_compile.wait(globals())
del async_compile

def call(args):
    a_1, b_1 = args
    args.clear()
    assert_size_stride(a_1, (2, 3), (3, 1))
    assert_size_stride(b_1, (3, 4), (4, 1))
    buf0 = empty_strided_cpu((2, 4), (4, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.mm(a_1, b_1, out=buf0)
    del a_1
    del b_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    a_1 = rand_strided((2, 3), (3, 1), device='cpu', dtype=torch.float32)
    b_1 = rand_strided((3, 4), (4, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([a_1, b_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
