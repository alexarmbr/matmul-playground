#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include "batched-gemm/helper.h"

// #define M 2048
// #define N 1024
// #define K 512

#define N 128
#define M 256
#define K 64

// #define N 16
// #define M 16
// #define K 16

// 8 tiles per block
// #define M_TILES_PER_BLOCK 2
// #define N_TILES_PER_BLOCK 4
#define M_TILES_PER_BLOCK 1
#define N_TILES_PER_BLOCK 1


// 16 by 16 tiles
#define TILE_DIM 16

#define WARP_SIZE 32

using namespace nvcuda;

// D = alpha * A * B + beta * C
__global__ void sgemm(half* A, half* B, half* C, half* D, float alpha, float beta)
{
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


int main(int argc, char **argv) {
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
    srand(1234);
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

    // printf("0th row of A: ");
    // for (int i = 0; i < K; i++)
    // {
    //   printf("%f, ", (float)A[i]);
    // }
    // printf("\n");
    // printf("0th row of B: ");
    // for (int i = 0; i < N; i++)
    // {
    //   printf("%f,", (float)B[i]);
    // }


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
          // D_host[m * N + n] = (half) acc;
          // D_host[m * N + n] = alpha * (half) acc + beta * C[m * N + n];
        }
      }
    
    float max_diff = 0.01;
    for (int i = 0; i < M * N; i++)
    {
      if (std::abs((float)D[i] - (float)D_host[i]) > max_diff)
      {
        printf("Mismatch at %d: %f vs %f\n", i, (float)D[i], (float)D_host[i]);
        break;
      }
    }

  }
    return 0;
  }