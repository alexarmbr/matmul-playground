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
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  const unsigned int warpsPerBlock = blockDim.x / WARP_SIZE; // WARP_SIZE should divide blockDim.x
  const unsigned int warpRowIdx = (threadIdx.y + blockIdx.y * blockDim.y) * TILE_DIM; // y threads are spaced out TILE_DIM units
  const unsigned int warpColIdx = (blockIdx.x * warpsPerBlock + warpIdx) * TILE_DIM; // 1 warp, 32 threads with consecutive threadIdx.x
  // compute a 16x16 tile of C
 
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
    // TODO double check this
    const unsigned int tileRowsPerBlock = 2;
    const unsigned int tileColsPerBlock = 4;
    const unsigned int yBlocks = N / (tileRowsPerBlock * TILE_DIM);
    const unsigned int xBlocks = M / (tileColsPerBlock * TILE_DIM);
    const unsigned int yTheadsPerBlock = tileRowsPerBlock;
    const unsigned int xThreadsPerBlock = WARP_SIZE * tileColsPerBlock;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yTheadsPerBlock);
    sgemm<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, dev_D, alpha, beta);
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(D, dev_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    
    if (check_on_cpu) {
      half D_host[M * N];

      for (int m = 0; m < M; m++)
      {
        for (int n = 0; n < N; n++)
        {
          
          float acc = 0.0f;
          for (int k = 0; k < K; k++)
          {
            acc += (float) (A[m * K + k] * B[k * N + n]);
          }
          D_host[m * N + n] = alpha * (half) acc + beta * C[m * N + n];
        }
      }
    }
    return 0;
  }