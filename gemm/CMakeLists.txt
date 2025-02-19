

cmake_minimum_required(VERSION 3.21)
project(gemm LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)


# GPU ARCH
set(CMAKE_CUDA_ARCHITECTURES 80)
if (DEFINED ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${ARCH})
endif()

# target
add_executable(sgemm_rtx_3090 sgemm_rtx_3090.cu)
target_compile_options(sgemm_rtx_3090 PRIVATE "--expt-relaxed-constexpr")

add_executable(cublas_sgemm cublas_sgemm.cu)
target_link_libraries(cublas_sgemm PRIVATE cublas)

# Cutlass Path
include_directories($ENV{CUTLASS_PATH})
