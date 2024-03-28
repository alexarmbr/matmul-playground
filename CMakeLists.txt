cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
enable_testing()
set(CMAKE_BUILD_TYPE Debug)

# turing arch, tesla t4
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# cuda include directories
include_directories(include)

add_executable(runner src/main.cu)
# target_compile_options(tensorcore_helloworld PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

