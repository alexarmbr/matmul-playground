#include <gtest/gtest.h>
#include "host_utils.cuh"
#include "kernels/device_utils.cuh"


template<unsigned int M,
unsigned int N,
typename T>
__global__ void loadFromGmemKernelWrapper(
    float* src,
    float* dst,
    const unsigned int src_stride,
    const unsigned int dst_stride
)
{
    tileMemcpy<M, N, T>(src, dst, src_stride, dst_stride);
}


// test the kernel
TEST(TestFp32Utils, TestLoadTileFromGmem)
{
    const unsigned int M = 64;
    const unsigned int N = 16;

    float* src_host = new float[M * N];
    float* dst_host = new float[M * N];
    float* src_device;
    float* dst_device;
    CUDA_CHECK(cudaMalloc(&src_device, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_device, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dst_device, 0, M * N * sizeof(float)));

    for (int i = 0; i < M * N; i++)
    {
        src_host[i] = i;
    }

    CUDA_CHECK(cudaMemcpy(src_device, src_host, M * N * sizeof(float), cudaMemcpyHostToDevice));
    const unsigned int yBlocks = 1;
    const unsigned int xBlocks = 1;
    const unsigned int yThreadsPerBlock = 1;
    const unsigned int xThreadsPerBlock = 32;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    
    printf("running kernel\n");
    loadFromGmemKernelWrapper<M, N, float>
    <<<gridDim, blockDim>>>(
        src_device,
        dst_device,
        N,
        N
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(dst_host, dst_device, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            if (dst_host[row * N + col] != src_host[row * N + col])
            {
                printf("Expected %f but got %f at row %d, col %d\n", src_host[row * N + col], dst_host[row * N + col], row, col);
                ASSERT_EQ(dst_host[row * N + col], src_host[row * N + col]);
            }
        }
    }
}

// test the kernel