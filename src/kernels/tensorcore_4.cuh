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

  constexpr unsigned int WM_dim = 16;
  constexpr unsigned int WN_dim = 8;
  constexpr unsigned int WK_dim = 8;
  const unsigned int laneIdx = threadIdx.x % 32;

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

  // load tile of C into shared memory
  const unsigned int C_block_tile_index = block_m * CD_stride + block_n;
  tileMemcpy<BM_dim, BN_dim, half>(C + C_block_tile_index, CD_shmem_blocktile, CD_stride, CD_shmem_stride);
  
  const unsigned int CD_warp_tile_index = warp_m * CD_shmem_stride + warp_n;
  uint32_t C_register[2];
  uint32_t* smem_ptr_C_;
  {
    const int fragment_row = laneIdx % WM_dim;
    const int offset = fragment_row * 4;
    smem_ptr_C = reinterpret_cast<uint32_t*>(CD_shmem_blocktile) + offset;
  }
  














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
      bool accumulate_C = (block_k == K - BK_dim) && (warp_k == BK_dim - WK_dim);

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
        N,
        accumulate_C
      );
    }
    __syncthreads();
  }
  
}


