#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"

using namespace nvcuda;

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim>
__global__ void kernel_1(half* A,
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

  // declare wmma fragments for caching matrix fragments at warp level
  wmma::fragment<wmma::matrix_a, WM_dim, WN_dim, WK_dim, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM_dim, WN_dim, WK_dim, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM_dim, WN_dim, WK_dim, half> c_frag;
  wmma::fragment<wmma::accumulator, WM_dim, WN_dim, WK_dim, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  
  for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  {

    for (unsigned int warp_k = 0; warp_k < BK_dim; warp_k += WK_dim)
    {
      const unsigned int A_tile_index = (block_m + warp_m) * A_stride + (block_k + warp_k);
      const unsigned int B_tile_index = (block_k + warp_k) * B_stride + (block_n + warp_n); 
      wmma::load_matrix_sync(a_frag, A + A_tile_index, A_stride);
      wmma::load_matrix_sync(b_frag, B + B_tile_index, B_stride);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }
  
  const unsigned int CD_tile_index = (block_m + warp_m) * CD_stride + (block_n + warp_n);
  wmma::load_matrix_sync(c_frag, C + CD_tile_index, CD_stride, wmma::mem_row_major);
  const half beta_ = (half) beta;
  for (int i = 0; i < c_frag.num_elements; i++)
  {
    c_frag.x[i] = alpha * acc_frag.x[i] + (float) (beta_ * c_frag.x[i]);
  }
  
  wmma::store_matrix_sync(D + CD_tile_index, c_frag, CD_stride, wmma::mem_row_major);
}

void kernel_1_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 2;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 1;
    constexpr unsigned int BM_dim = WM_dim * WARPS_PER_BLOCK_M;
    constexpr unsigned int BN_dim = WN_dim * WARPS_PER_BLOCK_N;
    constexpr unsigned int BK_dim = WK_dim * WARPS_PER_BLOCK_K; 
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    const unsigned int ThreadsM = 1;
    const unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_1
        <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>
        <<<gridDim, blockDim>>>(
            device_sgemm_params.A,
            device_sgemm_params.B,
            device_sgemm_params.C,
            device_sgemm_params.D,
            device_sgemm_params.alpha,
            device_sgemm_params.beta,
            M,
            N,
            K
        );
        timer.Stop();
    }
    double gflops_per_sec = timer.logKernelStats(M, N, K);
    std::cout << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    CUDA_CHECK(cudaPeekAtLastError());
}