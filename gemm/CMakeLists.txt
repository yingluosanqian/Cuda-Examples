

cmake_minimum_required(VERSION 3.21)
project(gemm LANGUAGES CXX CUDA)
enable_language(CUDA)

# GPU ARCH
set(CMAKE_CUDA_ARCHITECTURES 80)
if (DEFINED ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${ARCH})
endif()

add_executable(sgemm_A10 sgemm_A10.cu)
target_compile_options(sgemm_A10 PRIVATE "--expt-relaxed-constexpr")

# Cutlass Path
include_directories($ENV{CUTLASS_PATH})
