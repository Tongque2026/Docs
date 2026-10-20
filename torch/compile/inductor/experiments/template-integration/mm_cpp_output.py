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
_tensor_constant0 = None  # device(type='cpu') torch.float32 (16, 16) (16, 1) 7f584ef22250
constant0 = None  # device(type='cpu') torch.float32 (1, 16, 48) (768, 48, 1) 7f58436a0950


cpp_fused_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache-template-integration/tmpmh8ofxas/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"

#include "c10/util/Unroll.h"



template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum>
inline void kernel_micro_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    using Vectorized = at::vec::Vectorized<float>;
    using VectorizedIn = at::vec::Vectorized<float>;
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;

    Vectorized va;
    at::vec::VectorizedN<float, COLS> vb;
    at::vec::VectorizedN<float, ROWS*COLS> vc;

    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
        } else {
            vc[i] = Vectorized(0.0f);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);

    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;

        if constexpr (col == 0) {
            va = Vectorized(static_cast<float>(A[row * lda + k]));
        }

        if constexpr (row == 0) {
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
        }

        constexpr int idx = row * COLS + col;
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    for (int k = 0; k < K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        vc[i].store(C + row * ldc + col * VLEN);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

template <bool accum>
inline void kernel_micro_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    TORCH_CHECK(N % 48 == 0, "N dimension must be multiple of 48");
    TORCH_CHECK(K % 1 == 0, "K dimension must be multiple of 1");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += 8) {
        int64_t block_m = std::min<int64_t>(M - m, 8);
        for (int64_t n = 0; n < N; n += 48) {
            if (block_m == 8) {
                kernel_micro_gemm_kernel<8, 48, accum>(
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc
                );
            } else {
                switch (block_m) {
                case 7:
                    kernel_micro_gemm_kernel<7, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 6:
                    kernel_micro_gemm_kernel<6, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 5:
                    kernel_micro_gemm_kernel<5, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 4:
                    kernel_micro_gemm_kernel<4, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 3:
                    kernel_micro_gemm_kernel<3, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 2:
                    kernel_micro_gemm_kernel<2, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                case 1:
                    kernel_micro_gemm_kernel<1, 48, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported block_m: ", block_m);
                }
            }
        }
    }
}

extern "C" 
void kernel(const float* X, const float* W, float* Y)
{

    constexpr int64_t num_threads = 1;
    constexpr int64_t N = 16;
    constexpr int64_t K = 16;
    constexpr int64_t Mr = 8;
    constexpr int64_t Nr = 48;
    constexpr int64_t Kr = 1;
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
    constexpr int64_t M = static_cast<int64_t>(8L);
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
    constexpr int64_t Mt_blocks = 1;
    constexpr int64_t Nt_blocks = 1;
    constexpr int64_t Kt_blocks = 16;
    constexpr int64_t Mc_blocks = 1;
    constexpr int64_t Nc_blocks = 1;
    constexpr int64_t Kc_blocks = 16;
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_k_slices = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;

    // make sure all partitions are assigned
    TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 1 >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
    {
        const int tid = 0;
        const int64_t m_block_start = 0;
        const int64_t m_block_end = Mr_blocks;
        const int64_t n_block_start = 0;
        const int64_t n_block_end = Nr_blocks;
        const int64_t k_block_start = 0;
        const int64_t k_block_end = Kr_blocks;

        auto _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); auto local_acc_buf = _local_acc_buf.get();
        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * Mr;
            const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
            const int64_t m_size = m_end - m_start;
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                const int64_t n_start = nc * Nr;
                const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                const int64_t n_size = n_end - n_start;
                // NB: assume we pad N, nc_block_end won't exceed padded N here.
                const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
                if (_local_acc_buf == nullptr) { _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); local_acc_buf = _local_acc_buf.get(); }
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
                        if (kc == k_block_start) {
                            kernel_micro_gemm<static_cast<bool>(false)>(
                                &(X[static_cast<int64_t>(k_start + (16L*m_start))]),
                                &(W[static_cast<int64_t>((48L*k_start) + (768L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(48L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        } else {
                            kernel_micro_gemm<static_cast<bool>(true)>(
                                &(X[static_cast<int64_t>(k_start + (16L*m_start))]),
                                &(W[static_cast<int64_t>((48L*k_start) + (768L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(48L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        }
                    }
                }
                {
                    {
                        #pragma GCC ivdep
                        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(m_end + ((-1L)*m_start)); x0+=static_cast<int64_t>(1L))
                        {
                            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))); x1+=static_cast<int64_t>(16L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + (Nc_blocks*Nr*x0)), static_cast<int64_t>(16));
                                auto tmp1 = (tmp0);
                                tmp1.store(Y + static_cast<int64_t>(n_start + x1 + (16L*m_start) + (16L*x0)));
                            }
                            for(int64_t x1=static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))); x1<static_cast<int64_t>(n_end + ((-1L)*n_start)); x1+=(static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))) == 0 ? 1 : static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))))))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + (Nc_blocks*Nr*x0)), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))));
                                auto tmp1 = (tmp0);
                                tmp1.store(Y + static_cast<int64_t>(n_start + x1 + (16L*m_start) + (16L*x0)), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))));
                            }
                        }
                    }

                }
            }
        }

    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    x_1, = args
    args.clear()
    assert_size_stride(x_1, (8, 16), (16, 1))
    buf0 = empty_strided_cpu((8, 16), (16, 1), torch.float32)
    cpp_fused_0(x_1, constant0, buf0)
    del x_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global constant0
    constant0 = rand_strided((1, 16, 48), (768, 48, 1), device='cpu', dtype=torch.float32)
    x_1 = rand_strided((8, 16), (16, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([x_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
