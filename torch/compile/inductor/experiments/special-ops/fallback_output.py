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


cpp_fused_0 = async_compile.cpp_pybinding(['float*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-special-ops/tmp_mr7ifwu/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
            auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
            in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    x_1, = args
    args.clear()
    assert_size_stride(x_1, (3, ), (1, ))
    # Topologically Sorted Source Nodes: [], Original ATen: []
    buf0 = torch.ops.inductor_learning_examples.pair.default(x_1)
    del x_1
    buf1 = buf0[0]
    buf2 = buf0[1]
    del buf0
    buf3 = buf1; del buf1  # reuse
    cpp_fused_0(buf3)
    return (buf3, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    x_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([x_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
