#pragma once
#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"

using namespace nvcuda;

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int MMA_M_dim,
unsigned int MMA_N_dim,
unsigned int MMA_K_dim
>
__global__ void tensorcore_3(half* A,
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

  // declare wmma fragments for caching matrix fragments at warp level
  // each warp computes MMA_TILES_M * MMA_TILES_N tiles of size WM * WN in the output
  static_assert(WM_dim % MMA_M_dim == 0);
  static_assert(WN_dim % MMA_N_dim == 0);
  static_assert(WK_dim % MMA_K_dim == 0);
  constexpr unsigned int MMA_TILES_M = WM_dim / MMA_M_dim;
  constexpr unsigned int MMA_TILES_N = WN_dim / MMA_N_dim;
  constexpr unsigned int MMA_TILES_K = WK_dim / MMA_K_dim;
  wmma::fragment<wmma::matrix_a, MMA_M_dim, MMA_N_dim, MMA_K_dim, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, MMA_M_dim, MMA_N_dim, MMA_K_dim, half, wmma::row_major> b_frag[MMA_TILES_K][MMA_TILES_N];
  wmma::fragment<wmma::accumulator, MMA_M_dim, MMA_N_dim, MMA_K_dim, half> c_frag;
  wmma::fragment<wmma::accumulator, MMA_M_dim, MMA_N_dim, MMA_K_dim, float> acc_frag[MMA_TILES_M][MMA_TILES_N];
  for (unsigned int tile_m = 0; tile_m < MMA_TILES_M; tile_m++)
  {
    for (unsigned int tile_n = 0; tile_n < MMA_TILES_N; tile_n++)
    {
      wmma::fill_fragment(acc_frag[tile_m][tile_n], 0.0f);
    }
  }

  // load tile of C into shared memory ahead of time
  tileMemcpy<BM_dim, BN_dim, half>(C + block_m * CD_stride + block_n, CD_shmem_blocktile, CD_stride, CD_shmem_stride);
  
  // loop over shared memory block tiles along K dimension
  for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  {
    // load in current block tile of A,B along K dimension into shared memory
    tileMemcpy<BM_dim, BK_dim, half>(A + block_m * A_stride + block_k, A_shmem_blocktile, A_stride, A_shmem_stride);
    tileMemcpy<BK_dim, BN_dim, half>(B + block_k * B_stride + block_n, B_shmem_blocktile, B_stride, B_shmem_stride);
    
    __syncthreads();

    // warps move along k dimension within the block tile
    for (unsigned int warp_k = 0; warp_k < BK_dim; warp_k += WK_dim)
    {
      const unsigned int A_warp_index = warp_m * A_shmem_stride + warp_k;
      const unsigned int B_warp_index = warp_k * B_shmem_stride + warp_n;

      // each warp preloads the tiles of B that fall inside its warp tile
      for (unsigned int tile_k = 0; tile_k < MMA_TILES_K; tile_k++)
      {
        for (unsigned int tile_n = 0; tile_n < MMA_TILES_N; tile_n++)
        {
          const unsigned int B_tile_index = B_warp_index + (tile_k * MMA_K_dim * B_shmem_stride) + (tile_n * MMA_N_dim);
          wmma::load_matrix_sync(b_frag[tile_k][tile_n], B_shmem_blocktile + B_tile_index, B_shmem_stride);
        }
      }

      
      // outer product between mma tiles of B (already cached in registers by this point)
      // and tiles of A which we load from shmem
      // looping over tiles of A first allows us to minimize shmem traffic, each warp
      // loads each tile of A from shared memory only once
      for (unsigned int tile_k = 0; tile_k < MMA_TILES_K; tile_k++)
      {
        for (unsigned int tile_m = 0; tile_m < MMA_TILES_M; tile_m++)
        {
          const unsigned int A_tile_index = A_warp_index + (tile_m * MMA_M_dim * A_shmem_stride) + (tile_k * MMA_K_dim);
          wmma::load_matrix_sync(a_frag, A_shmem_blocktile + A_tile_index, A_shmem_stride);
          for (unsigned int tile_n = 0; tile_n < MMA_TILES_N; tile_n++)
          {
            wmma::mma_sync(acc_frag[tile_m][tile_n], a_frag, b_frag[tile_k][tile_n], acc_frag[tile_m][tile_n]);
          }
        }
      }

    }
    __syncthreads();
  }
  
  // loop over tiles of C,D and accumulate the final result from C, acc_frag into D
  const unsigned int CD_warp_index = warp_m * CD_shmem_stride + warp_n;
  const half beta_ = (half) beta;
  for (unsigned int tile_m = 0; tile_m < MMA_TILES_M; tile_m++)
  {
    for (unsigned int tile_n = 0; tile_n < MMA_TILES_N; tile_n++)
    {
      const unsigned int CD_tile_index = CD_warp_index + (tile_m * MMA_M_dim * CD_shmem_stride) + (tile_n * MMA_N_dim);
      wmma::load_matrix_sync(c_frag, CD_shmem_blocktile + CD_tile_index, CD_shmem_stride, wmma::mem_row_major);
      for (int i = 0; i < c_frag.num_elements; i++)
      {
        c_frag.x[i] = alpha * acc_frag[tile_m][tile_n].x[i] + (float) (beta_ * c_frag.x[i]);
      }
      wmma::store_matrix_sync(CD_shmem_blocktile + CD_tile_index, c_frag, CD_shmem_stride, wmma::mem_row_major);
    }
  }

  __syncthreads();
  tileMemcpy<BM_dim, BN_dim, half>(CD_shmem_blocktile, D + block_m * CD_stride + block_n, CD_shmem_stride, CD_stride);
}


