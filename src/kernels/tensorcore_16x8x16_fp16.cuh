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

    // B is 8x8
    half B_register[2];
    const unsigned int A_shared_stride_bytes = K * sizeof(half);
    const unsigned int B_shared_stride_bytes = N * sizeof(half);
    const unsigned int CD_shared_stride_bytes = N * sizeof(half);
    ldmatrix_n8k8(B_shared, B_register, B_shared_stride_bytes);

    // scale B by alpha
    half* B_register_half = reinterpret_cast<half*>(&B_register);
    B_register_half[0] *= alpha;
    B_register_half[1] *= alpha;

    // load A,C
    half A_register[4];
    half C_register[4];

    ldmatrix_m16n8(C_shared, C_register, CD_shared_stride_bytes);
    ldmatrix_m16n8(A_shared, A_register, A_shared_stride_bytes);

    C_register[0] *= beta;
    C_register[1] *= beta;
    C_register[2] *= beta;
    C_register[3] *= beta;
    
    // compute D
    mma_sync_m16n8k8(C_register, A_register, B_register, C_register);
    stmatrix_m16n8(D, C_register, CD_shared_stride_bytes);
}
