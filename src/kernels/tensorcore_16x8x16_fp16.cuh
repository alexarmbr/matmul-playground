#pragma once
#include <cuda.h>
#include "device_utils.cuh"

__global__ void tensorcore_m16n8k8(half* A,
    half* B,
    half* C,
    half* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{   
    // 1d block, 1 warp
    assert(blockDim.x == 32);
    assert(threadIdx.y == 0);
    assert(M == 16);
    assert(N == 8);
    assert(K == 8);

    __shared__ half A_shared[16 * 8];
    __shared__ half B_shared[8 * 8];
    __shared__ half C_shared[16 * 8];

    // load A and B into shared memory
    tileMemcpy<16, 8, half>(A, A_shared, K, 8);
    tileMemcpy<8, 8, half>(B, B_shared, N, 8);
    tileMemcpy<16, 8, half>(C, C_shared, N, 8);
    __syncthreads();

    mma_m16n8k8(A_shared, B_shared, C_shared, D, alpha, beta, 8, 8, 8, 8);
}
