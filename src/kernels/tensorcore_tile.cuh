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

    const unsigned int CD_stride = N;
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;

    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int BK_dim = 64;
    
    constexpr unsigned int WM_dim = 128;
    constexpr unsigned int WN_dim = 64;
    constexpr unsigned int WK_dim = 64;

    constexpr unsigned int MMA_M_dim = 16;
    constexpr unsigned int MMA_N_dim = 8;
    constexpr unsigned int MMA_K_dim = 8;

    static_assert(WM_dim % MMA_M_dim == 0);
    static_assert(WN_dim % MMA_N_dim == 0);
    static_assert(WK_dim % MMA_K_dim == 0);
    constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
    constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
    constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;

    static_assert(BM_dim % WM_dim == 0);
    static_assert(BN_dim % WN_dim == 0);
    static_assert(BK_dim % WK_dim == 0);
    constexpr unsigned int warp_tiles_per_block_n = BN_dim / WN_dim;
    constexpr unsigned int warp_tiles_per_block_m = BM_dim / WM_dim;
    constexpr unsigned int warp_tiles_per_block_k = BK_dim / WK_dim;
    
    const unsigned int warp_index = threadIdx.x / 32;
    const unsigned int warp_m_index = warp_index / warp_tiles_per_block_n;
    const unsigned int warp_n_index = warp_index % warp_tiles_per_block_n;
    const unsigned int warp_m = warp_m_index * WM_dim;
    const unsigned int warp_n = warp_n_index * WN_dim;
    
    // 1d block, 1 warp
    assert(blockDim.x == 64);
    assert(threadIdx.y == 0);
    assert(M == BM_dim);
    assert(N == BN_dim);
    assert(K == BK_dim);


    __shared__ half A_shared[BM_dim * BK_dim];
    __shared__ half B_shared[BK_dim * BN_dim];

    // load A and B into shared memory
    tileMemcpyTranspose<BM_dim, BK_dim>(A, A_shared, K * sizeof(half), M * sizeof(half));
    tileMemcpyTranspose<BK_dim, BN_dim>(B, B_shared, N * sizeof(half), K * sizeof(half));
    // __syncthreads();
    // if (threadIdx.x > 31)
    // {
    //     return;
    // }

    const unsigned int B_mma_stride_k = MMA_K_dim * MMA_N_dim;
    const unsigned int B_mma_stride_n = mma_tiles_per_warp_k * B_mma_stride_k;
    const unsigned int B_warp_stride_k = B_mma_stride_k * mma_tiles_per_warp_k;
    const unsigned int B_warp_stride_n = warp_tiles_per_block_k * B_warp_stride_k;

    const unsigned int A_mma_stride_m = MMA_M_dim * MMA_K_dim;
    const unsigned int A_mma_stride_k = mma_tiles_per_warp_m * A_mma_stride_m;
    
    const unsigned int CD_mma_stride_m = MMA_M_dim * MMA_N_dim;
    // const unsigned int CD_mma_stride_n = mma_tiles_per_warp_m * CD_mma_stride_m;
    
    const unsigned int thread_stride_bytes = 8 * sizeof(half);

    half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
    
    // this loops indices are in units of matrix elements
    for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
    {
        for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
        {
            const unsigned int mma_m_ind = mma_m / MMA_M_dim;
            const unsigned int mma_n_ind = mma_n / MMA_N_dim;

            // TODO add blocktile offset here when needed
            const unsigned int C_offset = (warp_m + mma_m) * CD_stride + (warp_n + mma_n);
            ldmatrix_m16n8_gmem(C + C_offset, C_register[mma_m_ind][mma_n_ind], CD_stride * sizeof(half));
            
            // scale C by beta
            C_register[mma_m_ind][mma_n_ind][0] *= beta;
            C_register[mma_m_ind][mma_n_ind][1] *= beta;
            C_register[mma_m_ind][mma_n_ind][2] *= beta;
            C_register[mma_m_ind][mma_n_ind][3] *= beta;
        }
    }
    
    // this loops indices are in units of mma tiles
    for (unsigned int mma_k_index = 0; mma_k_index < mma_tiles_per_warp_k; mma_k_index++)
    {
        for (unsigned int mma_m_index = 0; mma_m_index < mma_tiles_per_warp_m; mma_m_index++)
        {
            // load (m,k) from A
            half A_register[4];
            
            const unsigned int A_offset = ((A_mma_stride_m * mma_m_index) + warp_m) + A_mma_stride_k * mma_k_index;
            ldmatrix_m16n8(A_shared + A_offset, A_register, thread_stride_bytes);

            // scale A by alpha
            A_register[0] *= alpha;
            A_register[1] *= alpha;
            A_register[2] *= alpha;
            A_register[3] *= alpha;

            for (unsigned int mma_n_index = 0; mma_n_index < mma_tiles_per_warp_n; mma_n_index++)
            {
                // load (k, n) from B
                half B_register[2];
                // TODO something wrong with this index calculation
                const unsigned int B_offset = (B_mma_stride_k * mma_k_index) + (B_mma_stride_n * mma_n_index) + (B_warp_stride_n * warp_n_index);
                ldmatrix_n8k8(B_shared + B_offset, B_register, thread_stride_bytes);
                mma_sync_m16n8k8(C_register[mma_m_index][mma_n_index], A_register, B_register, C_register[mma_m_index][mma_n_index]);
            }
        }
    }

    // for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
    // {
    //     for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
    //     {
    //         const unsigned int D_offset = mma_m * MMA_M_dim * BN_dim + mma_n * MMA_N_dim;
    //         stmatrix_m16n8(D + D_offset, C_register[mma_m][mma_n], BN_dim * sizeof(half)); 
    //     }
    // }
    // this loops indices are in units of matrix elements
    for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
    {
        for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
        {
            const unsigned int mma_m_ind = mma_m / MMA_M_dim;
            const unsigned int mma_n_ind = mma_n / MMA_N_dim;

            const unsigned int D_offset = (warp_m + mma_m) * CD_stride + (warp_n + mma_n);
            stmatrix_m16n8(D + D_offset, C_register[mma_m_ind][mma_n_ind], BN_dim * sizeof(half)); 
        }
    }
    




}
