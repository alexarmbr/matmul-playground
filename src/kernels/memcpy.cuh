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
__global__ void memcpy(half* A,
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
  const unsigned int CD_stride = N;
  
  // top left coords of the block tile
  const unsigned int block_m = blockIdx.y * BM_dim;
  const unsigned int block_n = blockIdx.x * BN_dim;

  // declare shared memory tiles for caching matrix tiles at block level
  __shared__ half CD_shmem_blocktile[BM_dim * BN_dim];
  constexpr unsigned int CD_shmem_stride = BN_dim;

  // load tile of C ahead of time
  tileMemcpy<BM_dim, BN_dim, half>(C + block_m * CD_stride + block_n, CD_shmem_blocktile, CD_stride, CD_shmem_stride);
  __syncthreads();
  tileMemcpy<BM_dim, BN_dim, half>(CD_shmem_blocktile, D + block_m * CD_stride + block_n, CD_shmem_stride, CD_stride);
}


