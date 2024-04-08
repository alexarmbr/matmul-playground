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
__global__ void tensorcore_2(half* A,
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
  constexpr unsigned int WARPS_PER_BLOCK_M = BM_dim / WM_dim;
  constexpr unsigned int WARPS_PER_BLOCK_N = BN_dim / WN_dim;
  constexpr unsigned int WARP_SIZE = 32;
  
  // top left coords of the block tile
  const unsigned int block_m = blockIdx.y * BM_dim;
  const unsigned int block_n = blockIdx.x * BN_dim;
  
  // top left coords of a warp tile, relative to the block tile its in
  // TODO get rid of threadIdx.y
  const unsigned int warp_index = threadIdx.x / WARP_SIZE;
  const unsigned int warp_m = warp_index / WARPS_PER_BLOCK_N;
  const unsigned int warp_n = warp_index - (warp_m * WARPS_PER_BLOCK_M);

  __shared__ float A_shmem_blocktile[BK_dim * BN_dim * WK_dim * WN_dim];
  __shared__ float B_shmem_blocktile[BM_dim * BK_dim * WM_dim * WK_dim];
  __shared__ float CD_shmem_blocktile[BM_dim * BN_dim * WM_dim * WN_dim];

  // for (int block_k = 0; block_k )
  // tileMemcpy<BN, BM, half>(C + block_m * N + block_n, CD_shmem_blocktile, N);
  // tileMemcpy<BN, BN, half>(CD_shmem_blocktile, D + block_m * N + block_n)



}


