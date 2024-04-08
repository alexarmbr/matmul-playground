#include <gtest/gtest.h>
#include "host_utils.cuh"
#include "kernels/device_utils.cuh"


template<unsigned int TILE_ROWS,
unsigned int TILE_COLS>
__global__ void loadFromGmemKernelWrapper(
    float* A_gmem,
    float* A_shared,
    const unsigned int A_stride
)
{
    tileMemcpy<TILE_ROWS, TILE_COLS, float>(A_gmem, A_shared, A_stride, TILE_ROWS);
}


// test the kernel
TEST(TestFp32Utils, TestLoadTileFromGmem)
{
    const unsigned int M = 512;
    const unsigned int N = 256;
    const unsigned int TILE_ROWS = 16;
    const unsigned int TILE_COLS = 32;

    float* A_gmem_host = new float[M * N];
    float* A_shared_host = new float[TILE_ROWS * TILE_COLS];
    float* A_gmem_device;
    float* A_shared_device;
    CUDA_CHECK(cudaMalloc(&A_gmem_device, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_shared_device, TILE_ROWS * TILE_COLS * sizeof(float)));
    CUDA_CHECK(cudaMemset(A_shared_device, 0, TILE_ROWS * TILE_COLS * sizeof(float)));

    for (int i = 0; i < M * N; i++)
    {
        A_gmem_host[i] = i;
    }

    CUDA_CHECK(cudaMemcpy(A_gmem_device, A_gmem_host, M * N * sizeof(float), cudaMemcpyHostToDevice));
    const unsigned int yBlocks = 1;
    const unsigned int xBlocks = 1;
    const unsigned int yThreadsPerBlock = 1;
    const unsigned int xThreadsPerBlock = 32;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    
    loadFromGmemKernelWrapper<TILE_ROWS, TILE_COLS>
    <<<gridDim, blockDim>>>(
        A_gmem_device,
        A_shared_device,
        N
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(A_shared_host, A_shared_device, TILE_ROWS * TILE_COLS * sizeof(float), cudaMemcpyDeviceToHost));
    for (int row = 0; row < TILE_ROWS; row++)
    {
        for (int col = 0; col < TILE_COLS; col++)
        {
            if (A_shared_host[row * TILE_COLS + col] != A_gmem_host[row * N + col])
            {
                printf("Expected %f but got %f at row %d, col %d\n", A_gmem_host[row * N + col], A_shared_host[row * TILE_COLS + col], row, col);
                EXPECT_EQ(A_shared_host[row * TILE_COLS + col], A_gmem_host[row * N + col]);
            }
        }
    }
}

// test the kernel