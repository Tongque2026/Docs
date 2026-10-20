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


cpp_fused_0 = async_compile.cpp_pybinding(['const double*', 'const double*', 'double*', 'const int64_t', 'const int64_t'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-fxgraph-proof/run-vtvibxky/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const double* in_ptr0,
                       const double* in_ptr1,
                       double* out_ptr0,
                       const int64_t ks0,
                       const int64_t ks1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((ks0*ks1)), static_cast<int64_t>(16L)))); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::VectorizedN<double,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp1 = at::vec::VectorizedN<double,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
            auto tmp4 = static_cast<double>(2.0);
            auto tmp5 = at::vec::VectorizedN<double,2>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            tmp6.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
        }
        #pragma omp simd simdlen(8) 
        for(int64_t x0=static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((ks0*ks1)), static_cast<int64_t>(16L)))); x0<static_cast<int64_t>(ks0*ks1); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
            auto tmp1 = in_ptr1[static_cast<int64_t>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp3 = std::max(tmp2, decltype(tmp2)(0));
            auto tmp4 = static_cast<double>(2.0);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            out_ptr0[static_cast<int64_t>(x0)] = tmp5;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    x_1, y_1 = args
    args.clear()
    x_1_size = x_1.size()
    s0 = x_1_size[0]
    s1 = x_1_size[1]
    assert_size_stride(x_1, (s0, s1), (s1, 1))
    assert_size_stride(y_1, (s0, s1), (s1, 1))
    buf0 = empty_strided_cpu((s0, s1), (s1, 1), torch.float64)
    cpp_fused_0(x_1, y_1, buf0, s0, s1)
    del x_1
    del y_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    x_1 = rand_strided((2, 3), (3, 1), device='cpu', dtype=torch.float64)
    y_1 = rand_strided((2, 3), (3, 1), device='cpu', dtype=torch.float64)
    fn = lambda: call([x_1, y_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
