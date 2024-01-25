cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)

# googletest
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
enable_testing()

# turing arch, tesla t4
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# cuda include directories
include_directories(include)

add_executable(test_helpers test/test_helpers.cu)
target_link_libraries(test_helpers GTest::gtest_main)

add_executable(run_matmul src/matmul.cu)
target_link_libraries(run_matmul CUDA::cublasLt)