#include <cuda.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include "device_utils.cuh"
#include "structs_n_stuff.cuh"





using namespace cute;

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim>
__global__ void kernel_2_cute(half* A,
  half* B,
  half* C,
  half* D,
  const float alpha,
  const float beta,
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{

  Tensor A_gmem = make_tensor(A, make_shape(M, K), LayoutRight{});
  Tensor B_gmem = make_tensor(B, make_shape(K, N), LayoutRight{});
  Tensor C_gmem = make_tensor(C, make_shape(M, N), LayoutRight{});
  Tensor D_gmem = make_tensor(D, make_shape(M, N), LayoutRight{});

  constexpr unsigned int WM_dim = 16;
  constexpr unsigned int WN_dim = 8;
  constexpr unsigned int WK_dim = 8;

  // declare shared memory tiles for caching matrix tiles at block level
  __shared__ half A_shmem_[BM_dim * BK_dim];
  __shared__ half B_shmem_[BK_dim * BN_dim];

  auto A_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BK_dim>{});
  auto B_block_tile_shape = make_shape(Int<BK_dim>{}, Int<BN_dim>{});
  auto CD_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BN_dim>{});
  auto A_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WK_dim>{});
  auto B_warp_tile_shape = make_shape(Int<WK_dim>{}, Int<WN_dim>{});
  auto CD_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WN_dim>{});
  
  Tensor A_smem = make_tensor(make_smem_ptr(A_shmem_), A_block_tile_shape, LayoutRight{});
  Tensor B_smem = make_tensor(make_smem_ptr(B_shmem_), B_block_tile_shape, LayoutRight{});

  Tensor A_block_tiles = zipped_divide(A_gmem, A_block_tile_shape);
  Tensor B_block_tiles = zipped_divide(B_gmem, B_block_tile_shape);
  Tensor C_block_tiles = zipped_divide(C_gmem, CD_block_tile_shape);
  Tensor D_block_tiles = zipped_divide(D_gmem, CD_block_tile_shape);

  Tensor C_block_tile = C_block_tiles(make_coord(_,_), make_coord(blockIdx.y, blockIdx.x));
  Tensor C_warp_tile = zipped_divide(C_block_tile, CD_warp_tile_shape);
  
  half C_register[4];
  ldmatrix_m16n8_gmem(C_warp_tile.data(), C_register, N * sizeof(half));
  C_register[0] *= beta;
  C_register[1] *= beta;
  C_register[2] *= beta;
  C_register[3] *= beta;

  const unsigned int m_tiles = size<1,0>(A_block_tiles);
  const unsigned int k_tiles = size<1,1>(A_block_tiles);
  const unsigned int n_tiles = size<1,1>(B_block_tiles);
  assert(m_tiles == M / BM_dim);
  assert(k_tiles == K / BK_dim);
  assert(n_tiles == N / BN_dim);

  for (unsigned int k_block = 0; k_block < k_tiles; k_block++)
  {
    copy(A_block_tiles(make_coord(_,_), make_coord(blockIdx.y, k_block)), A_smem);
    copy(B_block_tiles(make_coord(_,_), make_coord(k_block, blockIdx.x)), B_smem);
    // Tensor A_warp_tiles 
    
    __syncthreads();


  }

  Tensor D_block_tile = D_block_tiles(make_coord(_,_), make_coord(blockIdx.y, blockIdx.x));
  Tensor D_warp_tile = zipped_divide(D_block_tile, CD_warp_tile_shape);
  stmatrix_m16n8(D_warp_tile.data(), C_register, N * sizeof(half));
}




void kernel_2_cute_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 8;
    constexpr unsigned int WK_dim = 8;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 4;
    constexpr unsigned int BM_dim = WM_dim * WARPS_PER_BLOCK_M;
    constexpr unsigned int BN_dim = WN_dim * WARPS_PER_BLOCK_N;
    constexpr unsigned int BK_dim = WK_dim * WARPS_PER_BLOCK_K;
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

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_2_cute
        <BM_dim, BN_dim, BK_dim>
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


