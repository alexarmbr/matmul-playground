#pragma once
#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"

using namespace nvcuda;

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim>
__global__ void tensorcore_4(half* A,
  half* B,
  half* C,
  half* D,
  const float alpha,
  const float beta,
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{
  static_assert(BM_dim % WM_dim == 0);
  static_assert(BN_dim % WN_dim == 0);
  constexpr unsigned int WARPS_PER_BLOCK_N = BN_dim / WN_dim;
  constexpr unsigned int WARP_SIZE = 32;
  constexpr unsigned int WM = 16;
  constexpr unsigned int WN = 8;
  constexpr unsigned int WK = 8;
  const unsigned int A_stride = K;
  const unsigned int B_stride = N;
  const unsigned int CD_stride = N;
  
  // top left coords of the block tile
  const unsigned int block_m = blockIdx.y * BM_dim;
  const unsigned int block_n = blockIdx.x * BN_dim;
  
  // top left coords of a warp tile, relative to the block tile its in
  const unsigned int warp_index = threadIdx.x / WARP_SIZE;
  const unsigned int warp_tile_m = warp_index / WARPS_PER_BLOCK_N;
  const unsigned int warp_tile_n = warp_index - (warp_tile_m * WARPS_PER_BLOCK_N);
  const unsigned int warp_m = warp_tile_m * WM_dim;
  const unsigned int warp_n = warp_tile_n * WN_dim;

  // declare shared memory tiles for caching matrix tiles at block level
  __shared__ half A_shmem_blocktile[BM_dim * BK_dim];
  __shared__ half B_shmem_blocktile[BK_dim * BN_dim];
  __shared__ half CD_shmem_blocktile[BM_dim * BN_dim];
  constexpr unsigned int A_shmem_stride = BK_dim;
  constexpr unsigned int B_shmem_stride = BN_dim;
  constexpr unsigned int CD_shmem_stride = BN_dim;

  // load tile of C ahead of time
  tileMemcpy<BM_dim, BN_dim, half>(C + block_m * CD_stride + block_n, CD_shmem_blocktile, CD_stride, CD_shmem_stride);
  
  for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  {
    // load in current tile of A,B along K dimension
    tileMemcpy<BM_dim, BK_dim, half>(A + block_m * A_stride + block_k, A_shmem_blocktile, A_stride, A_shmem_stride);
    tileMemcpy<BK_dim, BN_dim, half>(B + block_k * B_stride + block_n, B_shmem_blocktile, B_stride, B_shmem_stride);
    
    __syncthreads();

    for (unsigned int warp_k = 0; warp_k < BK_dim; warp_k += WK_dim)
    {
      const unsigned int A_tile_index = warp_m * A_shmem_stride + warp_k;
      const unsigned int B_tile_index = warp_k * B_shmem_stride + warp_n;

      mma_m16n8k8(
        A_shmem_blocktile + A_tile_index,
        B_shmem_blocktile + B_tile_index,
        CD_shmem_blocktile + CD_tile_index,
        D,
        alpha,
        beta,
        A_shmem_stride,
        B_shmem_stride,
        CD_shmem_stride,
        D_stride
      )

      // wmma::load_matrix_sync(a_frag, A_shmem_blocktile + A_tile_index, A_shmem_stride);
      // wmma::load_matrix_sync(b_frag, B_shmem_blocktile + B_tile_index, B_shmem_stride);
      // wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }
  
  // const unsigned int CD_tile_index = warp_m * CD_shmem_stride + warp_n;
  // wmma::load_matrix_sync(c_frag, CD_shmem_blocktile + CD_tile_index, CD_shmem_stride, wmma::mem_row_major);
  // const half beta_ = (half) beta;
  // for (int i = 0; i < c_frag.num_elements; i++)
  // {
  //   c_frag.x[i] = alpha * acc_frag.x[i] + (float) (beta_ * c_frag.x[i]);
  // }
  
  // wmma::store_matrix_sync(CD_shmem_blocktile + CD_tile_index, c_frag, CD_shmem_stride, wmma::mem_row_major);
  // __syncthreads();
  tileMemcpy<BM_dim, BN_dim, half>(CD_shmem_blocktile, D + block_m * CD_stride + block_n, CD_shmem_stride, CD_stride);
}


