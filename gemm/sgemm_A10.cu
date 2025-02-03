// clear && nvcc --expt-relaxed-constexpr -arch=sm_86 -I"../../cutlass/build_sm86/install/include" ./sgemm_A10.cu -o sgemm_A10

#include <cstdlib>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

template<
    typename ProblemShape, typename CtaShape,
    typename TA, typename SALayout, typename TiledCopyA,
    typename TB, typename SBLayout, typename TiledCopyB,
    typename TC, typename SCLayout, typename TiledMma
>
__global__
void gemm_device(
    ProblemShape problem_shape, CtaShape cta_shape,
    TA const* A, SALayout sA_layout, TiledCopyA tiled_copy_A,
    TB const* B, SBLayout sB_layout, TiledCopyB tiled_copy_B,
    TC const* C, SCLayout sC_layout, TiledMma tiled_mma
) {
    using namespace cute;


}


template<typename TA, typename TB, typename TC>
void gemm_tn(
    int m,
    int n,
    int k,
    TA const* A,
    TB const* B,
    TC const* C
) {
    using namespace cute;
    auto problem_shape = make_shape(m, n, k);
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto bP = Int<3>{};
    auto cta_shape = make_shape(bM, bN, bK);

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
    
    dim3 grid_dim(size(tiled_mma));
    dim3 block_dim(ceil_div(m, bM), ceil_div(n, bN), 1);
    gemm_device<<<grid_dim, block_dim>>>(
        problem_shape, cta_shape,
        A, sA_layout, tiled_copy_A,
        B, sB_layout, tiled_copy_B,
        C, sC_layout, tiled_mma
    );
}


template<typename TA, typename TB, typename TC>
void gemm(
    int m,
    int n,
    int k,
    TA const* A,
    TB const* B,
    TC const* C
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

    using TA = float;
    using TB = float;
    using TC = float;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    for (int i = 0; i < m * k; i++) {
        h_A[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int i = 0; i < n * k; i++) {
        h_B[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int i = 0; i < m * n; i++) {
        h_C[i] = 0;
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

    return 0;
}