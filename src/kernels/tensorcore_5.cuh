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
unsigned int WK_dim>
__global__ void tensorcore_5(half* A,
  half* B,
  half* C,
  half* D,
  const float alpha,
  const float beta,
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{

  static_assert(WK_dim == BK_dim, "warptiling in k dimension not implemented");
  constexpr unsigned int MMA_M_dim = 16;
  constexpr unsigned int MMA_N_dim = 8;
  constexpr unsigned int MMA_K_dim = 8;

  static_assert(WM_dim % MMA_M_dim == 0);
  static_assert(WN_dim % MMA_N_dim == 0);
  static_assert(WK_dim % MMA_K_dim == 0);
  constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
  constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
  constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
  constexpr unsigned int warps_per_block_n = BN_dim / WN_dim;
  
  constexpr unsigned int WARP_SIZE = 32;
  const unsigned int A_stride_elements = K;
  const unsigned int B_stride_elements = N;
  const unsigned int CD_stride_elements = N;
  // const unsigned int A_stride_bytes = K * sizeof(half);
  // const unsigned int B_stride_bytes = N * sizeof(half);
  // const unsigned int CD_stride_bytes = N * sizeof(half);
  
  // top left coords of the block tile
  const unsigned int block_m = blockIdx.y * BM_dim;
  const unsigned int block_n = blockIdx.x * BN_dim;
  
  // top left coords of a warp tile, relative to the block tile its in
  const unsigned int warp_index = threadIdx.x / WARP_SIZE;
  const unsigned int warp_tile_m = warp_index / warps_per_block_n;
  const unsigned int warp_tile_n = warp_index % warps_per_block_n;
  const unsigned int warp_m = warp_tile_m * WM_dim;
  const unsigned int warp_n = warp_tile_n * WN_dim;

  // declare shared memory tiles for caching matrix tiles at block level
  __shared__ half A_shmem_blocktile[BM_dim * BK_dim];
  __shared__ half B_shmem_blocktile[BK_dim * BN_dim];
  constexpr unsigned int A_shmem_stride_elements = BK_dim;
  constexpr unsigned int B_shmem_stride_elements = BN_dim;
  constexpr unsigned int CD_shmem_stride_elements = BN_dim;
  // constexpr unsigned int A_shmem_stride_bytes = A_shmem_stride_elements * sizeof(half);
  // constexpr unsigned int B_shmem_stride_bytes = B_shmem_stride_elements * sizeof(half);
  // constexpr unsigned int CD_shmem_stride_bytes = BN_dim * sizeof(half);

  // declare register storage to hold fragments of C which we will accumulate into
  // for current params:
  //   (cuda-gdb) p blockDim
  // $5 = {x = 512, y = 1, z = 1}
  // (cuda-gdb) p gridDim 
  // $6 = {x = 32, y = 32, z = 1}
  // (cuda-gdb) p mma_tiles_per_warp_m
  // $7 = 2
  // (cuda-gdb) p mma_tiles_per_warp_n
  // $8 = 4
  // 512 threads * 2m * 4n * 2registers/thread = 8192 registers/thread
  half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];

  // this loops indices are in units of matrix elements
  for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
  {
      for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
      {
          const unsigned int mma_m_ind = mma_m / MMA_M_dim;
          const unsigned int mma_n_ind = mma_n / MMA_N_dim;
          const unsigned int C_offset = (block_m + warp_m + mma_m) * CD_stride_elements + (block_n + warp_n + mma_n);
          ldmatrix_m16n8_gmem(C + C_offset, C_register[mma_m_ind][mma_n_ind], CD_stride_elements * sizeof(half));
          
          // scale C by beta
          C_register[mma_m_ind][mma_n_ind][0] *= beta;
          C_register[mma_m_ind][mma_n_ind][1] *= beta;
          C_register[mma_m_ind][mma_n_ind][2] *= beta;
          C_register[mma_m_ind][mma_n_ind][3] *= beta;
      }
  }

  // half A_register[4];
  // half B_register[2];
  // for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  // {
  //   // load in current tile of A,B along K dimension
  //   const unsigned int A_block_tile_index = block_m * A_stride_elements + block_k;
  //   const unsigned int B_block_tile_index = block_k * B_stride_elements + block_n;
  //   tileMemcpy<BM_dim, BK_dim, half>(A + A_block_tile_index, A_shmem_blocktile, A_stride_elements, A_shmem_stride_elements);
  //   tileMemcpy<BK_dim, BN_dim, half>(B + B_block_tile_index, B_shmem_blocktile, B_stride_elements, B_shmem_stride_elements);
    
  //   __syncthreads();

  //   for (unsigned int warp_k = 0; warp_k < BK_dim; warp_k += MMA_K_dim)
  //   {
  //     const unsigned int A_tile_index = warp_m * A_shmem_stride_elements + warp_k;
  //     const unsigned int B_tile_index = warp_k * B_shmem_stride_elements + warp_n;
  //     ldmatrix_m16n8(A_shmem_blocktile + A_tile_index, A_register, A_shmem_stride_bytes);
  //     ldmatrix_n8k8(B_shmem_blocktile + B_tile_index, B_register, B_shmem_stride_bytes);
  //     B_register[0] *= alpha;
  //     B_register[1] *= alpha;
  //     mma_sync_m16n8k8(
  //       C_register,
  //       A_register,
  //       B_register,
  //       C_register
  //     );
  //   }
  //   __syncthreads();
  // }

  for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
  {
      for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
      {
          const unsigned int mma_m_ind = mma_m / MMA_M_dim;
          const unsigned int mma_n_ind = mma_n / MMA_N_dim;

          const unsigned int D_offset = (block_m + warp_m + mma_m) * CD_stride_elements + (block_n + warp_n + mma_n);
          stmatrix_m16n8(D + D_offset, C_register[mma_m_ind][mma_n_ind], CD_stride_elements * sizeof(half)); 
      }
  }
}


