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


cpp_fused_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-external-fallback/tmpldlfxkvg/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(2L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(3L))
            {
                alignas(16) float tmp0[16*16];
                at::vec::transpose_mxn<float,static_cast<int64_t>(3L),static_cast<int64_t>(2L)>(in_ptr0 + static_cast<int64_t>(x0 + (2L*x1)), static_cast<int64_t>(2L), tmp0, static_cast<int64_t>(3L));
                for (long x0_inner = 0; x0_inner < static_cast<int64_t>(2L); x0_inner++)
                {
                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(3L*x0_inner), static_cast<int64_t>(3L));
                    tmp1.store(out_ptr0 + static_cast<int64_t>(x1 + (3L*x0) + (3L*x0_inner)), static_cast<int64_t>(3L));
                }
            }
        }
    }
}
''')


cpp_fused_1 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-external-fallback/tmpldlfxkvg/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(2L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(3L))
            {
                alignas(16) float tmp1[16*16];
                for (long x0_inner = 0; x0_inner < static_cast<int64_t>(2L); x0_inner++)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (3L*x0) + (3L*x0_inner)), static_cast<int64_t>(3L));
                    tmp0.store(tmp1 + static_cast<int64_t>(3L*x0_inner), static_cast<int64_t>(3L));
                }
                at::vec::transpose_mxn<float,static_cast<int64_t>(2L),static_cast<int64_t>(3L)>(tmp1, static_cast<int64_t>(3L), out_ptr0 + static_cast<int64_t>(x0 + (2L*x1)), static_cast<int64_t>(2L));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(6L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = out_ptr0[static_cast<int64_t>(x0)];
            auto tmp1 = static_cast<float>(2.0);
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            out_ptr1[static_cast<int64_t>(x0)] = tmp2;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    x_1, = args
    args.clear()
    assert_size_stride(x_1, (2, 3), (1, 2))
    buf0 = empty_strided_cpu((2, 3), (3, 1), torch.float32)
    cpp_fused_0(x_1, buf0)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    torch.ops.inductor_external_lesson.bump_copy.default(buf0, 3.0)
    buf4 = empty_strided_cpu((2, 3), (1, 2), torch.float32)
    cpp_fused_1(buf0, x_1, buf4)
    del buf0
    del x_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    x_1 = rand_strided((2, 3), (1, 2), device='cpu', dtype=torch.float32)
    fn = lambda: call([x_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
