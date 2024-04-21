#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim>
__global__ void
kernel_4(half* A,
  half* B,
  half* C,
  half* D,
  const float alpha,
  const float beta,
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{

  constexpr unsigned int MMA_M_dim = 16;
  constexpr unsigned int MMA_N_dim = 8;
  constexpr unsigned int MMA_K_dim = 8;

  static_assert(WM_dim % MMA_M_dim == 0);
  static_assert(WN_dim % MMA_N_dim == 0);
  static_assert(WK_dim % MMA_K_dim == 0);
  constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
  constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
  constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;

  const unsigned int warp_tiles_per_block_k = BK_dim / WK_dim;
  const unsigned int warp_tiles_per_block_m = BM_dim / WM_dim;
  const unsigned int warp_tiles_per_block_n = BN_dim / WN_dim;
  
  constexpr unsigned int WARP_SIZE = 32;
  const unsigned int A_stride_elements = K;
  const unsigned int B_stride_elements = N;
  const unsigned int CD_stride_elements = N;
  const unsigned int A_stride_bytes = K * sizeof(half);
  const unsigned int B_stride_bytes = N * sizeof(half);
  // const unsigned int CD_stride_bytes = N * sizeof(half);
  
  // top left coords of the block tile
  const unsigned int block_m = blockIdx.y * BM_dim;
  const unsigned int block_n = blockIdx.x * BN_dim;
  
  // top left coords of a warp tile, relative to the block tile its in
  const unsigned int warp_index = threadIdx.x / WARP_SIZE;
  const unsigned int warp_m_ind = warp_index / warp_tiles_per_block_n;
  const unsigned int warp_n_ind = warp_index % warp_tiles_per_block_n;
  const unsigned int warp_m = warp_m_ind * WM_dim;
  const unsigned int warp_n = warp_n_ind * WN_dim;

  // declare shared memory tiles for caching matrix tiles at block level
  extern __shared__ half shmem[];
  half* A_shmem_blocktile = shmem;
  half* B_shmem_blocktile = &shmem[BM_dim * BK_dim];
  
  constexpr unsigned int A_shmem_transpose_col_stride_elements = MMA_M_dim * MMA_K_dim;
  constexpr unsigned int A_shmem_transpose_row_stride_elements = A_shmem_transpose_col_stride_elements * mma_tiles_per_warp_m * warp_tiles_per_block_m;
  constexpr unsigned int B_shmem_transpose_col_stride_elements = MMA_N_dim * MMA_K_dim;
  constexpr unsigned int B_shmem_transpose_row_stride_elements = B_shmem_transpose_col_stride_elements * mma_tiles_per_warp_k * warp_tiles_per_block_k;

  // declare register storage to hold fragments of C which we will accumulate into
  half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];

  // this loops indices are in units of matrix elements
  for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
  {
      for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
      {
          const unsigned int mma_m_ind = mma_m / MMA_M_dim;
          const unsigned int mma_n_ind = mma_n / MMA_N_dim;
          const unsigned int C_offset = (block_m + warp_m + mma_m) * CD_stride_elements + (block_n + warp_n + mma_n);
          ldmatrix_m16n8_gmem(C + C_offset, C_register[mma_m_ind][mma_n_ind], CD_stride_elements * sizeof(half));
          
          // scale C by beta
          C_register[mma_m_ind][mma_n_ind][0] *= beta;
          C_register[mma_m_ind][mma_n_ind][1] *= beta;
          C_register[mma_m_ind][mma_n_ind][2] *= beta;
          C_register[mma_m_ind][mma_n_ind][3] *= beta;
      }
  }

  for (unsigned int block_k = 0; block_k < K; block_k += BK_dim)
  {
    // load in current tile of A,B along K dimension
    const unsigned int A_block_tile_index = block_m * A_stride_elements + block_k;
    const unsigned int B_block_tile_index = block_k * B_stride_elements + block_n;
    tileMemcpyTranspose<BM_dim, BK_dim>(A + A_block_tile_index, A_shmem_blocktile, A_stride_bytes, BM_dim * sizeof(half));
    tileMemcpyTranspose<BK_dim, BN_dim>(B + B_block_tile_index, B_shmem_blocktile, B_stride_bytes, BK_dim * sizeof(half));
    
    __syncthreads();
    // preload tiles of A into registers
    
    for (unsigned int warp_k_ind = 0; warp_k_ind < warp_tiles_per_block_k; warp_k_ind++)
    {
      half A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4];
      for (unsigned int mma_m_ind = 0; mma_m_ind < mma_tiles_per_warp_m; mma_m_ind++)
      {
        for (unsigned int mma_k_ind = 0; mma_k_ind < mma_tiles_per_warp_k; mma_k_ind++)
        {
          const unsigned int mma_tile_row_ind = warp_k_ind * mma_tiles_per_warp_k + mma_k_ind;
          const unsigned int mma_tile_col_ind = warp_m_ind * mma_tiles_per_warp_m + mma_m_ind;
          const unsigned int A_shmem_offset = mma_tile_row_ind * A_shmem_transpose_row_stride_elements +
          mma_tile_col_ind * A_shmem_transpose_col_stride_elements;
          ldmatrix_m16n8(A_shmem_blocktile + A_shmem_offset, A_register[mma_m_ind][mma_k_ind], sizeof(float4));
        }
      }

      // load one tile of B at a time, and take outer product between this tile and
      // entire warp tile of A
      half B_register[2];
      for (unsigned int mma_k_ind = 0; mma_k_ind < mma_tiles_per_warp_k; mma_k_ind++)
      {
        for (unsigned int mma_n_ind = 0; mma_n_ind < mma_tiles_per_warp_n; mma_n_ind++)
        {
          const unsigned int mma_tile_row_ind = warp_n_ind * mma_tiles_per_warp_n + mma_n_ind;
          const unsigned int mma_tile_col_ind = warp_k_ind * mma_tiles_per_warp_k + mma_k_ind;
          const unsigned int B_shmem_offset = mma_tile_row_ind * B_shmem_transpose_row_stride_elements +
          mma_tile_col_ind * B_shmem_transpose_col_stride_elements;
          
          ldmatrix_n8k8(B_shmem_blocktile + B_shmem_offset, B_register, sizeof(float4));
          B_register[0] *= alpha;
          B_register[1] *= alpha;
          for (unsigned int mma_m_ind = 0; mma_m_ind < mma_tiles_per_warp_m; mma_m_ind++)
          {
            mma_sync_m16n8k8(
              C_register[mma_m_ind][mma_n_ind],
              A_register[mma_m_ind][mma_k_ind],
              B_register,
              C_register[mma_m_ind][mma_n_ind]
            );
          }
        }
      }
  }
    __syncthreads();
  }

  for (unsigned int mma_m = 0; mma_m < WM_dim; mma_m+=MMA_M_dim)
  {
      for (unsigned int mma_n = 0; mma_n < WN_dim; mma_n+=MMA_N_dim)
      {
          const unsigned int mma_m_ind = mma_m / MMA_M_dim;
          const unsigned int mma_n_ind = mma_n / MMA_N_dim;

          const unsigned int D_offset = (block_m + warp_m + mma_m) * CD_stride_elements + (block_n + warp_n + mma_n);
          stmatrix_m16n8(D + D_offset, C_register[mma_m_ind][mma_n_ind], CD_stride_elements * sizeof(half)); 
      }
  }
}

void kernel_4_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    
    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int BK_dim = 64;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 2;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 4;

    constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
    constexpr unsigned int WN_dim = BM_dim / WARPS_PER_BLOCK_N;
    constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;

    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    assert(M % BM_dim == 0);
    assert(N % BN_dim == 0);
    assert(K % BK_dim == 0);
    
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    const unsigned int ThreadsM = 1;
    const unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_4<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    65536)); // set shared memory limit to 64KB which is maximum for sm_75

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_4
        <BM_dim, BN_dim, BK_dim,
        WM_dim, WN_dim, WK_dim>
        <<<gridDim, blockDim, shmem_bytes>>>(
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


