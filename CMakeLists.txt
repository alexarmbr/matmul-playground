cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)

# googletest
find_package(GTest REQUIRED)
enable_testing()

# turing arch, tesla t4
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include_directories(include)

add_executable(test_helpers test/test_helpers.cu)
target_link_libraries(test_helpers GTest::gtest_main)
