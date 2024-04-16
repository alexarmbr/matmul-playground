#pragma once
#include <cuda.h>
#include "device_utils.cuh"

__global__ void tensorcore_m64n64k64(half* A,
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
    assert(M == 64);
    assert(N == 64);
    assert(K == 64);

    __shared__ half A_shared[64 * 64];
    __shared__ half B_shared[64 * 64];
    __shared__ half C_shared[64 * 64];

    // load A and B into shared memory
    tileMemcpySwizzle(A, A_shared, K * sizeof(half), K * sizeof(half), 64);
    tileMemcpySwizzle(B, B_shared, N * sizeof(half), N * sizeof(half), 64);
    tileMemcpySwizzle(C, C_shared, N * sizeof(half), N * sizeof(half), 64);
    __syncthreads();

    // B is 8x8
    // half B_register[2];
    // const unsigned int A_shared_stride_bytes = K * sizeof(half);
    // const unsigned int B_shared_stride_bytes = N * sizeof(half);
    // const unsigned int CD_shared_stride_bytes = N * sizeof(half);
    // ldmatrix_n8k8(B_shared, B_register, B_shared_stride_bytes);

    // // scale B by alpha
    // half* B_register_half = reinterpret_cast<half*>(&B_register);
    // B_register_half[0] *= alpha;
    // B_register_half[1] *= alpha;

    // // load A,C
    // half A_register[4];
    // half C_register[4];

    // ldmatrix_m16n8(C_shared, C_register, CD_shared_stride_bytes);
    // ldmatrix_m16n8(A_shared, A_register, A_shared_stride_bytes);

    // C_register[0] *= beta;
    // C_register[1] *= beta;
    // C_register[2] *= beta;
    // C_register[3] *= beta;
    
    // // compute D
    // mma_sync_m16n8k8(C_register, A_register, B_register, C_register);
    // stmatrix_m16n8(D, C_register, CD_shared_stride_bytes);
}
