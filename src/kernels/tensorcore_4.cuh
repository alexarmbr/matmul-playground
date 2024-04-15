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

  static_assert(BM_dim % WM_dim == 0);
  static_assert(BN_dim % WN_dim == 0);
  constexpr unsigned int WARPS_PER_BLOCK_N = BN_dim / WN_dim;
  constexpr unsigned int WARP_SIZE = 32;
  const unsigned int A_stride_elements = K;
  const unsigned int B_stride_elements = N;
  const unsigned int CD_stride_elements = N;
  // const unsigned int A_stride_bytes = K * sizeof(half);
  // const unsigned int B_stride_bytes = N * sizeof(half);
  const unsigned int CD_stride_bytes = N * sizeof(half);
  
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
  constexpr unsigned int A_shmem_stride_elements = BK_dim;
  constexpr unsigned int B_shmem_stride_elements = BN_dim;
  constexpr unsigned int CD_shmem_stride_elements = BN_dim;
  constexpr unsigned int A_shmem_stride_bytes = A_shmem_stride_elements * sizeof(half);
  constexpr unsigned int B_shmem_stride_bytes = B_shmem_stride_elements * sizeof(half);
  constexpr unsigned int CD_shmem_stride_bytes = BN_dim * sizeof(half);

  // load tile of C into shared memory
  const unsigned int C_block_tile_index = block_m * CD_stride_elements + block_n;
  tileMemcpy<BM_dim, BN_dim, half>(C + C_block_tile_index, CD_shmem_blocktile, CD_stride_elements, CD_shmem_stride_elements);

  __syncthreads();
  // load fragment of C into registers and scale by beta
  const unsigned int CD_warp_tile_index = warp_m * CD_shmem_stride_elements + warp_n;
  half C_register[4];
  ldmatrix_m16n8(CD_shmem_blocktile + CD_warp_tile_index, C_register, CD_shmem_stride_bytes);
  C_register[0] *= beta;
  C_register[1] *= beta;
  C_register[2] *= beta;
  C_register[3] *= beta;

  half A_register[4];
  half B_register[2];
  for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  {
    // load in current tile of A,B along K dimension
    const unsigned int A_block_tile_index = block_m * A_stride_elements + block_k;
    const unsigned int B_block_tile_index = block_k * B_stride_elements + block_n;
    tileMemcpy<BM_dim, BK_dim, half>(A + A_block_tile_index, A_shmem_blocktile, A_stride_elements, A_shmem_stride_elements);
    tileMemcpy<BK_dim, BN_dim, half>(B + B_block_tile_index, B_shmem_blocktile, B_stride_elements, B_shmem_stride_elements);
    
    __syncthreads();

    for (unsigned int warp_k = 0; warp_k < BK_dim; warp_k += WK_dim)
    {
      const unsigned int A_tile_index = warp_m * A_shmem_stride_elements + warp_k;
      const unsigned int B_tile_index = warp_k * B_shmem_stride_elements + warp_n;
      ldmatrix_m16n8(A_shmem_blocktile + A_tile_index, A_register, A_shmem_stride_bytes);
      ldmatrix_n8k8(B_shmem_blocktile + B_tile_index, B_register, B_shmem_stride_bytes);
      B_register[0] *= alpha;
      B_register[1] *= alpha;
      mma_sync_m16n8k8(
        C_register,
        A_register,
        B_register,
        C_register
      );
    }
    __syncthreads();
  }

  const unsigned int D_gmem_index = (block_m + warp_m) * CD_stride_elements + block_n + warp_n;
  stmatrix_m16n8(D + D_gmem_index, C_register, CD_stride_bytes);
}


