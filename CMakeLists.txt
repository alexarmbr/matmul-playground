cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)


enable_testing()
set(CMAKE_BUILD_TYPE Debug)

# turing arch, tesla t4
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# file(GLOB SOURCES "src/kernel*.cu")
set(SOURCES src/kernel7.cu src/kernel6.cu src/kernel5.cu src/kernel8.cu)

add_library(kernels ${SOURCES})
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=cg>)

target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
# target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
target_include_directories(kernels PRIVATE /tmp/cutlass/include)

add_library(cublas_launch src/cublas_benchmark.cu)
target_link_libraries(cublas_launch CUDA::cublas)

add_executable(runner src/main.cu)
target_link_libraries(runner PUBLIC kernels cublas_launch)