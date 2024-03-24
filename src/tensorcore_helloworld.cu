#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include "batched-gemm/helper.h"

#define M 2048
#define N 1024
#define K 512

// 8 tiles per block
#define M_TILES_PER_BLOCK 2
#define N_TILES_PER_BLOCK 4

// 16 by 16 tiles
#define TILE_DIM 16

#define WARP_SIZE 32

// D = alpha * A * B + beta * C
__global__ void sgemm(half* A, half* B, half* C, half* D, float alpha, float beta)
{
  const unsigned int laneIdx = threadIdx.x % WARP_SIZE;
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  const unsigned int warpsPerBlock = blockDim.x / WARP_SIZE; // WARP_SIZE should divide blockDim.x
  const unsigned int warpRowIdx = (threadIdx.y + blockIdx.y * blockDim.y) * TILE_DIM; // y threads are spaced out TILE_DIM units
  const unsigned int warpColIdx = (blockIdx.x * warpsPerBlock + warpIdx) * TILE_DIM; // 1 warp, 32 threads with consecutive threadIdx.x
  
  // compute a 16x16 tile of C
  // TODO are right flags being passed to nvcc to make sure debug symbols are generated for
  // device code?


  const unsigned int tileCol = threadIdx.x % TILE_DIM;
  const unsigned int tileRow = laneIdx / TILE_DIM;
  unsigned int i = (warpRowIdx + tileRow) * N + (warpColIdx + tileCol);
  const half eps = 0.1;
  for (int step = 0; step < TILE_DIM / (WARP_SIZE / TILE_DIM); step++)
  {
    const half c = C[i];
    D[i] = c + eps;
    i += (2 * N);
  }
 
}


int main(int argc, char **argv) {
  std::cout << "hellooo" << std::endl;

    bool check_on_cpu = true;

    // setup

    half *A, *B, *C, *D;
    A = (half *)malloc(M * K * sizeof(half));
    B = (half *)malloc(K * N * sizeof(half));
    C = (half *)malloc(M * N * sizeof(half));
    D = (half *)malloc(M * N * sizeof(half));
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
    const unsigned int yBlocks = M / (M_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int xBlocks = N / (N_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int yThreadsPerBlock = M_TILES_PER_BLOCK;
    const unsigned int xThreadsPerBlock = WARP_SIZE * N_TILES_PER_BLOCK;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    sgemm<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, dev_D, alpha, beta);
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(D, dev_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++)
    {
      if (D[i] != C[i] + (half) 0.1)
      {
        printf("error at index %d, expected %f, got %f\n", i, (float) (C[i] + (half) 0.1), (float) D[i]);
        return 1;
      }
    }



      
    // if (check_on_cpu) {
    //   half D_host[M * N];

    //   for (int m = 0; m < M; m++)
    //   {
    //     for (int n = 0; n < N; n++)
    //     {
          
    //       float acc = 0.0f;
    //       for (int k = 0; k < K; k++)
    //       {
    //         acc += (float) (A[m * K + k] * B[k * N + n]);
    //       }
    //       D_host[m * N + n] = alpha * (half) acc + beta * C[m * N + n];
    //     }
    //   }
    // }
    return 0;
  }