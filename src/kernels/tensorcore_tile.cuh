#pragma once
#include <cuda.h>
#include "device_utils.cuh"

__global__ void tensorcore_tile(half* A,
    half* B,
    half* C,
    half* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{   
    constexpr unsigned int BM_ = 32;
    constexpr unsigned int MN_ = 16;
    constexpr unsigned int BK_ = 16;
    constexpr unsigned int WM_ = 16;
    constexpr unsigned int WN_ = 8;
    constexpr unsigned int WK_ = 8;

    
    // 1d block, 1 warp
    assert(blockDim.x == 32);
    assert(threadIdx.y == 0);
    assert(M == BM_);
    assert(N == BN_);
    assert(K == BK_);

    __shared__ half A_shared[BM_ * BK_];
    __shared__ half B_shared[BK_ * BN_];
    __shared__ half C_shared[BM_ * BN_];

    // load A and B into shared memory
    tileMemcpyTranspose<M_, K_>(A, A_shared, K * sizeof(half), M * sizeof(half));
    tileMemcpyTranspose<N_, K_>(B, B_shared, N * sizeof(half), K * sizeof(half));
    __syncthreads();

    const unsigned int B_warp_stride_m = WK_ * WN_ * sizeof(half);
    const unsigned int B_warp_stride_n = (BK_ / WK_) * B_warp_stride_m;
    const unsigned int A_warp_stride_m = WM_ * WK_ * sizeof(half);
    const unsigned int B_warp_stride_n = (BM_ / WM_) * A_warp_stride_m; 

    for (unsigned int warp_k = 0; warp_k < BK_; warp_k+=WK_)
    {
        for (unsigned int warp_m = 0; warp_m < BM_; warp_m+=WM_)
        {
            for (unsigned int warp_n = 0; warp_n < BN_; warp_n+=WN_)
            {
                half B_register[2];
                half A_register[4];
                const unsigned int thread_stride_bytes = 8 * sizeof(half);
                ldmatrix_n8k8(B_shared)
            }
        }
    }





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
