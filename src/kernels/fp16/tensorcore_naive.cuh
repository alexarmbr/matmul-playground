#pragma once
#include <cuda.h>
#include <mma.h>

// #define N 128
// #define M 256
// #define K 64

// // 8 tiles per block
// #define M_TILES_PER_BLOCK 2
// #define N_TILES_PER_BLOCK 4

// // 16 by 16 tiles
// #define TILE_DIM 16

using namespace nvcuda;

template <const unsigned int M_TILES_PER_BLOCK,
const unsigned int N_TILES_PER_BLOCK,
const unsigned int TILE_DIM>
__global__ void tensorcore_naive_sgemm(half* A,
  half* B,
  half* C,
  half* D,
  const float alpha,
  const float beta,
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{
  constexpr unsigned int WARP_SIZE = 32;
  const unsigned int laneIdx = threadIdx.x % WARP_SIZE;
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  const unsigned int warpsPerBlock = blockDim.x / WARP_SIZE; // WARP_SIZE should divide blockDim.x
  const unsigned int warpRowIdx = (threadIdx.y + blockIdx.y * blockDim.y) * TILE_DIM; // y threads are spaced out TILE_DIM units
  const unsigned int warpColIdx = (blockIdx.x * warpsPerBlock + warpIdx) * TILE_DIM; // 1 warp, 32 threads with consecutive threadIdx.x
  
  wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, half> c_frag;
  wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);
  
  // iterate over 16x16 tiles of A,B along the K dimension
  unsigned int k_idx = 0;
  while (k_idx < K)
  {
    // row * stride + col
    const unsigned int B_tile_index = k_idx * N + warpColIdx;
    const unsigned int A_tile_index = warpRowIdx * K + k_idx;
    
    // arguments are (fragment, pointer to source memory, stride of source memory)
    wmma::load_matrix_sync(a_frag, A + A_tile_index, K);
    wmma::load_matrix_sync(b_frag, B + B_tile_index, N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    k_idx += TILE_DIM;
  }

  const unsigned int CD_tile_index = warpRowIdx * N + warpColIdx;
  
  // load in current tile of C
  wmma::load_matrix_sync(c_frag, C + CD_tile_index, N, wmma::mem_row_major);
  
  const half beta_ = (half) beta;
  for (int i = 0; i < c_frag.num_elements; i++)
  {
    c_frag.x[i] = alpha * acc_frag.x[i] + (float) (beta_ * c_frag.x[i]);
  }
  
  wmma::store_matrix_sync(D + CD_tile_index, c_frag, N, wmma::mem_row_major);
}


