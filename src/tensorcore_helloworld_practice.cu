#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include "batched-gemm/helper.h"

#define M 2048
#define N 1024
#define K 512

// this should be num SMs
// #define NUM_BLOCKS 4

// 8 tiles per block
#define M_TILES_PER_BLOCK 2
#define N_TILES_PER_BLOCK 4

// 16 by 16 tiles
#define TILE_DIM 16

// how many tiles do we need to cover the M x N matrix?
#define M_NUM_TILES M / (M_TILES_PER_BLOCK * TILE_DIM)
#define N_NUM_TILES N / (N_TILES_PER_BLOCK * TILE_DIM)

#define WARP_SIZE 32

// D = alpha * A * B + beta * C
__global__ void sgemm(half* A, half* B, half* C, half* D, float alpha, float beta)
{
  const unsigned int block_start_ind = (blockIdx.y * M_TILES_PER_BLOCK * N) + (blockIdx.x * N_TILES_PER_BLOCK);
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  const unsigned int warpRow = warpIdx / N_TILES_PER_BLOCK;
  const unsigned int warpCol = warpIdx % N_TILES_PER_BLOCK;
  block_start_ind += (warpRow * TILE_DIM * N) + (warpCol * TILE_DIM); // row offset + column offset

  // to test whether tiling logic and index calculation works
  // iterate over this 16 by 16 tile, add 1, and store each value in D
}


int main(int argc, char **argv) {

    bool check_on_cpu = true;

    // setup
    half A[M * K], B[K * N], C[M * N], D[M * N];
    half alpha = 0.3;
    half beta = 0.7;

    half *dev_A, *dev_B, *dev_C, *dev_D;
    CUDA_CHECK(cudaMalloc((void **)&dev_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_D, M * N * sizeof(half)));

    // fill matrices with random elements
    for (int i = 0; i < M * N; i++) {
      C[i] = (half)(rand() % 10);
    }
    for (int i = 0; i < K * N; i++)
    {
      B[i] = (half)(rand() % 10);
    }
    for (int i = 0; i < M * K; i++)
    {
      A[i] = (half)(rand() % 10);
    }

    CUDA_CHECK(cudaMemcpy(dev_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_C, C, M * N * sizeof(half), cudaMemcpyHostToDevice));

    // launch kernel here
    dim3 gridDim(N_NUM_TILES, M_NUM_TILES);
    dim3 blockDim(N_TILES_PER_BLOCK * WARP_SIZE, M_TILES_PER_BLOCK);
    sgemm<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, dev_D, alpha, beta);
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(D, dev_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    
    if (check_on_cpu) {
      // check whether each value in D is 1 greater than the corresponding value in C
      for (int i = 0; i < M * N; i++) {
        assert(D[i] == C[i] + 1);
      }
    }
    return 0;
  }