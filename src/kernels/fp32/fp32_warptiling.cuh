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
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim>
__global__ void fp32_Warptiling_Sgemm(float* A,
    float* B,
    float* C,
    float* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{

    __shared__ float A_shmem_blocktile[BM * BK];
    __shared__ float B_shmem_blocktile[BK * BM];
    __shared__ float DC_shmem_blocktile[BM * BN];

    // calculate this thread blocks row index for matrices A,D,C
    const unsigned int blockIndexM = blockIdx.y * BM_dim;
    
    // calculate this thread blocks column index for matrices A,D,B
    const unsigned int blockIndexN = blockIdx.x * BN_dim;
    
    // iterate over tiles along the K dimension
    for (unsigned int blockStartK = 0; blockStartK < K; blockStartK += BK_dim)
    {
        loadFromGmem(A_shmem_blocktile, )
        // todo: how to optimize the outermost block tiling without worrying about the rest?

    }








}