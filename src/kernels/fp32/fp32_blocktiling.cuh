#pragma once
#include <cuda.h>
#include "device_utils.cuh"

// {BM/BN/BK}_dim - dimension of blocktile per M/N/K (matrix dimensions). We must be able to fit 3
// tiles in shared memory: BMxBN for C, BMxBK for B, BKxBN for B, so shared memory capacity 
// per SM is the limiting factor here. 1 thread block handles each tile.
//
// WM/WN/WK - dimensions of warp tiles per BM/BN/BK (block tile dimensions). 1 warp handles each tile
template<unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim>
__global__ void fp32_Blocktiled_Sgemm(float* A,
    float* B,
    float* C,
    float* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{
    static_assert(BK_dim % BM_dim == 0, "BK must be divisible by BM");
    static_assert(BK_dim % BN_dim == 0, "BK must be divisible by BM");
    
    // matrix strides
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int CD_stride = N;
    
    // declare shared memory tile, D/C matrices can share
    __shared__ float A_shmem_blocktile[BM_dim * BK_dim];
    __shared__ float B_shmem_blocktile[BK_dim * BM_dim];
    __shared__ float CD_shmem_blocktile[BM_dim * BN_dim];

    // calculate this thread blocks row index for matrices A,D,C
    const unsigned int block_m = blockIdx.y * BM_dim;
    
    // calculate this thread blocks column index for matrices A,D,B
    const unsigned int block_n = blockIdx.x * BN_dim;

    // calculate the row/col that this thread is responsible for relative to the thread block / shmem tiles
    const unsigned int thread_m = threadIdx.x / BN_dim;
    const unsigned int thread_n = threadIdx.x - (thread_m * BN_dim);
    
    // iterate over tiles along the K dimension
    float acc = 0.0f;
    for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
    {
        // load tiles of A from global memory to shared memory
        tileMemcpy<BM_dim, BK_dim>(A + (block_m * A_stride + block_k), A_shmem_blocktile, A_stride);
        tileMemcpy<BK_dim, BM_dim>(B + (block_k * B_stride + block_n), B_shmem_blocktile, B_stride);
        __syncthreads();

        // BN * BM threads in a block
        // we know BK % BN = 0 and BK % BM = 0
        for (unsigned int thread_k = 0; thread_k < BK_dim; thread_k += 1)
        {
            acc += B_shmem_blocktile[thread_k * BM_dim + thread_n] * A_shmem_blocktile[thread_m * BK_dim + thread_k];
        }
    }
    
    tileMemcpy<BM_dim, BN_dim>(C + (block_m * CD_stride) + block_n, CD_shmem_blocktile, CD_stride);
    CD_shmem_blocktile[thread_m * BN_dim + thread_n] = acc * alpha + CD_shmem_blocktile[thread_m * BN_dim + thread_n] * beta;
    tileMemcpy<BM_dim, BN_dim>(CD_shmem_blocktile, D + (block_m * CD_stride) + block_n, CD_stride);
}