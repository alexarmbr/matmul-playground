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
    constexpr unsigned int BM_ = 128;
    constexpr unsigned int BN_ = 128;
    constexpr unsigned int BK_ = 64;
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
    // __shared__ half C_shared[BM_ * BN_];

    // load A and B into shared memory
    tileMemcpyTranspose<BM_, BK_>(A, A_shared, K * sizeof(half), M * sizeof(half));
    tileMemcpyTranspose<BK_, BN_>(B, B_shared, N * sizeof(half), K * sizeof(half));
    // tileMemcpyTranspose<BM_, BN_>(C, C_shared, N * sizeof(half), M * sizeof(half));
    __syncthreads();
    
    constexpr unsigned int warp_tiles_per_block_k = (BK_ / WK_);
    constexpr unsigned int warp_tiles_per_block_m = (BM_ / WM_);
    constexpr unsigned int warp_tiles_per_block_n = (BN_ / WN_);

    const unsigned int B_warp_stride_k = WK_ * WN_;
    const unsigned int B_warp_stride_n = warp_tiles_per_block_k * B_warp_stride_k;
    const unsigned int A_warp_stride_m = WM_ * WK_;
    const unsigned int A_warp_stride_k = warp_tiles_per_block_m * A_warp_stride_m;
    const unsigned int CD_warp_stride_m = WM_ * WN_;
    const unsigned int CD_warp_stride_n = warp_tiles_per_block_m * CD_warp_stride_m;
    const unsigned int thread_stride_bytes = 8 * sizeof(half);

    half C_register[warp_tiles_per_block_m][warp_tiles_per_block_n][4];
    for (unsigned int warp_m = 0; warp_m < warp_tiles_per_block_m; warp_m++)
    {
        for (unsigned int warp_n = 0; warp_n < warp_tiles_per_block_n; warp_n++)
        {

            const unsigned int C_offset = warp_m * WM_ * BN_ + warp_n * WN_;
            ldmatrix_m16n8_gmem(C + C_offset, C_register[warp_m][warp_n], BN_ * sizeof(half));
            
            // scale C by beta
            C_register[warp_m][warp_n][0] *= beta;
            C_register[warp_m][warp_n][1] *= beta;
            C_register[warp_m][warp_n][2] *= beta;
            C_register[warp_m][warp_n][3] *= beta;
        }
    }
    
    for (unsigned int warp_k = 0; warp_k < warp_tiles_per_block_k; warp_k++)
    {
        for (unsigned int warp_m = 0; warp_m < warp_tiles_per_block_m; warp_m++)
        {
            // load (m,k) from A
            half A_register[4];
            const unsigned int A_offset = A_warp_stride_m *  warp_m + A_warp_stride_k * warp_k;
            ldmatrix_m16n8(A_shared + A_offset, A_register, thread_stride_bytes);

            // scale A by alpha
            A_register[0] *= alpha;
            A_register[1] *= alpha;
            A_register[2] *= alpha;
            A_register[3] *= alpha;

            for (unsigned int warp_n = 0; warp_n < warp_tiles_per_block_n; warp_n++)
            {
                // load (k, n) from B
                half B_register[2];
                const unsigned int B_offset = B_warp_stride_k * warp_k + B_warp_stride_n * warp_n;
                ldmatrix_n8k8(B_shared + B_offset, B_register, thread_stride_bytes);
                mma_sync_m16n8k8(C_register[warp_m][warp_n], A_register, B_register, C_register[warp_m][warp_n]);
            }
        }
    }

    for (unsigned int warp_m = 0; warp_m < warp_tiles_per_block_m; warp_m++)
    {
        for (unsigned int warp_n = 0; warp_n < warp_tiles_per_block_n; warp_n++)
        {
            const unsigned int D_offset = warp_m * WM_ * BN_ + warp_n * WN_;
            stmatrix_m16n8(D + D_offset, C_register[warp_m][warp_n], BN_ * sizeof(half)); 
        }
    }

}
