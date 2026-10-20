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


cpp_fused_0 = async_compile.cpp_pybinding(['const int64_t*', 'bool*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-special-ops/tmp540vskv2/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       bool* out_ptr0)
{
    {
        auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
        auto tmp1 = static_cast<int64_t>(3);
        auto tmp2 = tmp0 < tmp1;
        out_ptr0[static_cast<int64_t>(0L)] = tmp2;
    }
}
''')


cpp_fused_1 = async_compile.cpp_pybinding(['const float*', 'const int64_t*', 'float*', 'int64_t*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-special-ops/tmp540vskv2/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const int64_t* in_ptr1,
                       float* out_ptr0,
                       int64_t* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
            auto tmp1 = static_cast<float>(1.0);
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            out_ptr0[static_cast<int64_t>(x0)] = tmp2;
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<int64_t>(0L)];
        auto tmp1 = static_cast<int64_t>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr1[static_cast<int64_t>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    i_1, x_1 = args
    args.clear()
    assert_size_stride(i_1, (), ())
    assert_size_stride(x_1, (3, ), (1, ))
    buf0 = [None] * 2
    buf0[0] = i_1
    buf0[1] = x_1
    while True:
        # subgraph: while_loop_cond_graph_0
        while_loop_cond_graph_0_i_1 = buf0[0]
        while_loop_cond_graph_0_x_1 = buf0[1]
        while_loop_cond_graph_0_buf0 = empty_strided_cpu((), (), torch.bool)
        cpp_fused_0(while_loop_cond_graph_0_i_1, while_loop_cond_graph_0_buf0)
        del while_loop_cond_graph_0_i_1
        buf0_cond_result = while_loop_cond_graph_0_buf0
        if not buf0_cond_result.item(): break
        # subgraph: while_loop_body_graph_0
        while_loop_body_graph_0_i_1 = buf0[0]
        while_loop_body_graph_0_x_1 = buf0[1]
        while_loop_body_graph_0_buf0 = empty_strided_cpu((3, ), (1, ), torch.float32)
        while_loop_body_graph_0_buf1 = empty_strided_cpu((), (), torch.int64)
        cpp_fused_1(while_loop_body_graph_0_x_1, while_loop_body_graph_0_i_1, while_loop_body_graph_0_buf0, while_loop_body_graph_0_buf1)
        del while_loop_body_graph_0_i_1
        del while_loop_body_graph_0_x_1
        buf0[0] = while_loop_body_graph_0_buf1
        buf0[1] = while_loop_body_graph_0_buf0
    del i_1
    del x_1
    buf1 = buf0[0]
    buf2 = buf0[1]
    return (buf1, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    i_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    x_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([i_1, x_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
