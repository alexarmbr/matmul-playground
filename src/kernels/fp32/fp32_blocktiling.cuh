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
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int CD_stride = N;
    __shared__ float A_shmem_blocktile[BM_dim * BK_dim];
    __shared__ float B_shmem_blocktile[BK_dim * BM_dim];
    __shared__ float DC_shmem_blocktile[BM_dim * BN_dim];

    // calculate this thread blocks row index for matrices A,D,C
    const unsigned int blockIndexM = blockIdx.y * BM_dim;
    
    // calculate this thread blocks column index for matrices A,D,B
    const unsigned int blockIndexN = blockIdx.x * BN_dim;
    
    // iterate over tiles along the K dimension
    for (unsigned int blockStartK = 0; blockStartK < K; blockStartK += BK_dim)
    {
        // load tiles of A from global memory to shared memory
        loadFromGmem<BM_dim, BK_dim>(A + (blockIndexM * A_stride + blockStartK), A_shmem_blocktile, A_stride);
        loadFromGmem<BK_dim, BM_dim>(B + (blockStartK * B_stride + blockIndexN), B_shmem_blocktile, B_stride);

        // BN * BM threads in a block
        // we know BK % BN = 0 and BK % BM = 0
        

    }








}