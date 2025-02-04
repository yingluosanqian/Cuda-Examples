

cmake_minimum_required(VERSION 3.21)
project(gemm LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(sgemm_A10 sgemm_A10.cu)
target_compile_options(sgemm_A10 PRIVATE "--expt-relaxed-constexpr" "-arch=sm_86")

include_directories(/mnt/workspace/cutlass/build_sm86/install/include)
