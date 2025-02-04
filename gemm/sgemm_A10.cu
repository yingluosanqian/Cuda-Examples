// clear && nvcc --expt-relaxed-constexpr -arch=sm_86 -I"/mnt/workspace/cutlass/build_sm86/install/include" ./sgemm_A10.cu -o sgemm_A10

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

template<
    typename ProblemShape, typename CtaShape,
    typename TA, typename StrideA, typename SALayout, typename TiledCopyA,
    typename TB, typename StrideB, typename SBLayout, typename TiledCopyB,
    typename TC, typename StrideC, typename SCLayout, typename TiledMma
>
__global__
void gemm_device(
    ProblemShape problem_shape, CtaShape cta_shape,
    TA const* A, StrideA stride_a, SALayout sA_layout, TiledCopyA tiled_copy_A,
    TB const* B, StrideB stride_b, SBLayout sB_layout, TiledCopyB tiled_copy_B,
    TC      * C, StrideC stride_c, SCLayout sC_layout, TiledMma tiled_mma
) {
    using namespace cute;

    auto mA = make_tensor(make_gmem_ptr(A), select<0, 2>(problem_shape), stride_a); // (m, k)
    auto mB = make_tensor(make_gmem_ptr(B), select<1, 2>(problem_shape), stride_b); // (n, k)
    auto mC = make_tensor(make_gmem_ptr(C), select<0, 1>(problem_shape), stride_c); // (m, n)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    auto gA = local_tile(mA, cta_shape, cta_coord, Step<_1, X, _1>{}); // (bM, bK, k_pipe)
    auto gB = local_tile(mB, cta_shape, cta_coord, Step<X, _1, _1>{}); // (bN, bK, k_pipe)
    auto gC = local_tile(mC, cta_shape, cta_coord, Step<_1, _1, X>{}); // (bM, bN)

    __shared__ TA smemA[cosize_v<SALayout>];
    __shared__ TB smemB[cosize_v<SBLayout>];
    auto sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (bM, bK, bP)
    auto sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (bN, bK, bP)

    auto thr_copy_a = tiled_copy_A.get_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA); // (cpy, cpyM, cpyK, k_pipe)
    auto tAsA = thr_copy_a.partition_D(sA); // (cpy, cpyN, cpyK, bP)
    
    auto thr_copy_b = tiled_copy_B.get_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB); // (cpy, cpyM, cpyK, k_pipe)
    auto tBsB = thr_copy_b.partition_D(sB); // (cpy, cpyN, cpyK, bP)

    // Smem Prefetch
    int k_tile_count = size<3>(tAgA);
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_next = 0;
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
        copy(tiled_copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(tiled_copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        if (--k_tile_count > 0)
            k_tile_next++;
    }

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA); // (mma, mmaM, mmaK, bP)
    auto tCsB = thr_mma.partition_B(sB); // (mma, mmaN, mmaK, bP)
    auto tCgC = thr_mma.partition_C(gC); // (mma, mmaM, mmaN)
    auto tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0)); // (mma, mmaM, mmaK)
    auto tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0)); // (mma, mmaN, mmaK)
    auto tCrC = thr_mma.make_fragment_C(tCgC); // (mma, mmaM, mmaN)

    // Register Prefetch
    auto smem_pipe_read = 0;
    auto smem_pipe_write = K_PIPE_MAX - 1;
    auto tCsA_p = tCsA(_, _, _, smem_pipe_read); // (mma, mmaM, mmaK)
    auto tCsB_p = tCsB(_, _, _, smem_pipe_read); // (mma, mmaN, mmaK)

    auto K_BLOCK_MAX = size<2>(tCrA);
    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
        copy(tCsA_p(_, _, 0), tCrA(_, _, 0));
        copy(tCsB_p(_, _, 0), tCrB(_, _, 0));
    }

    // Mainloop
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {
            // gmem -> smem
            if (k_block == 0) {
                copy(tiled_copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(tiled_copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();
                if (--k_tile_count > 0)
                    k_tile_next++;
            }

            // smem -> rmem
            if (k_block == K_BLOCK_MAX - 1) {
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read++;
                smem_pipe_read = smem_pipe_read == K_PIPE_MAX ? 0 : smem_pipe_read;
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                tCsB_p = tCsB(_, _, _, smem_pipe_read);
            }
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(tCsA_p(_, _, k_block_next), tCrA(_, _, k_block_next));
            copy(tCsB_p(_, _, k_block_next), tCrB(_, _, k_block_next));

            // ffma
            gemm(tCrC, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // Epilogue
    axpby(1.0, tCrC, 0.0, tCgC);
}


template<typename TA, typename TB, typename TC>
void gemm_tn(
    int m,
    int n,
    int k,
    TA const* A,
    TB const* B,
    TC * C
) {
    using namespace cute;
    auto problem_shape = make_shape(m, n, k);
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto bP = Int<3>{};
    auto cta_shape = make_shape(bM, bN, bK);

    auto stride_a = make_stride(k, Int<1>{});
    auto stride_b = make_stride(k, Int<1>{});
    auto stride_c = make_stride(n, Int<1>{});

    auto sA_layout = make_layout(
        make_shape(bM, bK, bP),
        make_stride(Int<1>{}, bM + Int<4>{}, bK * (bM + Int<4>{}))
    );
    auto sB_layout = make_layout(
        make_shape(bN, bK, bP),
        make_stride(Int<1>{}, bN + Int<4>{}, bK * (bN + Int<4>{}))
    );
    auto sC_layout = make_layout(
        make_shape(bM, bN),
        make_stride(Int<1>{}, bM)
    );

    auto tiled_copy_A = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _1>>{}
    );
    auto tiled_copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _1>>{}
    );
    auto tiled_mma = make_tiled_mma(
        UniversalFMA<TC, TA, TB>{},
        Layout<Shape<_16, _16, _1>, Stride<_1, _16, _256>>{}
    );
    
    dim3 grid_dim(ceil_div(m, bM), ceil_div(n, bN), 1);
    dim3 block_dim(size(tiled_mma));
    gemm_device<<<grid_dim, block_dim>>>(
        problem_shape, cta_shape,
        A, stride_a, sA_layout, tiled_copy_A,
        B, stride_b, sB_layout, tiled_copy_B,
        C, stride_c, sC_layout, tiled_mma
    );
}


template<typename TA, typename TB, typename TC>
void gemm(
    int m,
    int n,
    int k,
    TA const* A,
    TB const* B,
    TC * C
) {
    gemm_tn(m, n, k, A, B, C);
}


int main(int argc, char** argv) {
    int m = 5120, n = 5120, k = 4096;

    if (argc >= 2) {
        sscanf(argv[1], "%d", &m);
    }
    if (argc >= 3) {
        sscanf(argv[2], "%d", &n);
    }
    if (argc >= 4) {
        sscanf(argv[3], "%d", &k);
    }

    char validation_char = 'N';
    if (argc >= 5) {
        sscanf(argv[4], "%c", &validation_char);
    }
    bool validation = validation_char == 'Y' ? true : false;

    using TA = float;
    using TB = float;
    using TC = float;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);
    thrust::host_vector<TC> h_validation(m * n);

    if (validation) {
        for (int i = 0; i < m * k; i++) {
            h_A[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
        }
        for (int i = 0; i < n * k; i++) {
            h_B[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
        }
        for (int i = 0; i < m * n; i++) {
            h_C[i] = h_validation[i] = 0;
        }
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    gemm(
        n,
        m,
        k,
        d_A.data().get(),
        d_B.data().get(),
        d_C.data().get()
    );
    thrust::host_vector<TC> cute_result = d_C;

    if (validation) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float C = 0;
                for (int x = 0; x < k; x++) {
                    C += h_A[i * k + x] * h_B[j * k + x];
                }
                h_validation[i * n + j] = C;
            }
        }
        bool same = true;
        for (int i = 0; i < m * n; i++) {
            if (fabs(cute_result[i] - h_validation[i]) > 1e-5) {
                printf("i: %d, result: %f, ref: %f\n", i, cute_result[i], h_validation[i]);
                same = false;
                break;
            }
        }
        assert(same && "Computation result error.");
        printf("Pass.\n");
    }

    return 0;
}