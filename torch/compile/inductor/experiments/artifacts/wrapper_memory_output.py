# AOT ID: ['0_inference']
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


cpp_fused_relu_0 = async_compile.cpp_pybinding(['float*'], '''
#include "/tmp/torchinductor_root/tmpotvt3ghz/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
        }
    }
}
''')


cpp_fused_sin_1 = async_compile.cpp_pybinding(['float*'], '''
#include "/tmp/torchinductor_root/tmpotvt3ghz/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp1 = tmp0.sin();
            tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 8), (8, 1))
    assert_size_stride(arg1_1, (8, 8), (8, 1))
    assert_size_stride(arg2_1, (8, 8), (8, 1))
    buf0 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
    # Topologically Sorted Source Nodes: [first], Original ATen: [aten.mm]
    extern_kernels.mm(arg1_1, arg0_1, out=buf0)
    del arg0_1
    del arg1_1
    buf1 = buf0; del buf0  # reuse
    cpp_fused_relu_0(buf1)
    buf2 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
    # Topologically Sorted Source Nodes: [activated, second], Original ATen: [aten.relu, aten.mm]
    extern_kernels.mm(buf1, arg2_1, out=buf2)
    del arg2_1
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused_sin_1(buf3)
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((8, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((8, 8), (8, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
