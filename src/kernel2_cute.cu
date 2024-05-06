#include <cuda.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include "device_utils.cuh"
#include "structs_n_stuff.cuh"
#include "cute_utils.cuh"

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
  constexpr unsigned int WM_dim = 16;
  constexpr unsigned int WN_dim = 8;
  constexpr unsigned int WK_dim = 8;

  auto A_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BK_dim>{});
  auto B_block_tile_shape = make_shape(Int<BK_dim>{}, Int<BN_dim>{});
  auto CD_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BN_dim>{});
  auto A_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WK_dim>{});
  auto B_warp_tile_shape = make_shape(Int<WK_dim>{}, Int<WN_dim>{});
  auto CD_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WN_dim>{});

  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;
  const unsigned int block_m = blockIdx.y;
  const unsigned int block_n = blockIdx.x;

  __shared__ half A_smem_[BM_dim * BK_dim];
  __shared__ half B_smem_[BK_dim * BN_dim];

  half A_register[4];
  half B_register[2];
  half C_register[4];

  Tensor A_gmem = make_tensor(A, make_shape(M, K), LayoutRight{});
  Tensor B_gmem = make_tensor(B, make_shape(K, N), LayoutRight{});
  Tensor C_gmem = make_tensor(C, make_shape(M, N), LayoutRight{});
  Tensor D_gmem = make_tensor(D, make_shape(M, N), LayoutRight{});

  Tensor A_smem = make_tensor(make_smem_ptr(A_smem_), A_block_tile_shape, LayoutRight{});
  Tensor B_smem = make_tensor(make_smem_ptr(B_smem_), B_block_tile_shape, LayoutRight{});

  // block tile each matrix
  Tensor A_block_tiles = zipped_divide(A_gmem, A_block_tile_shape);
  Tensor B_block_tiles = zipped_divide(B_gmem, B_block_tile_shape);
  Tensor C_block_tiles = zipped_divide(C_gmem, CD_block_tile_shape);
  Tensor D_block_tiles = zipped_divide(D_gmem, CD_block_tile_shape);
  
  // create warp tiles for a,b inside of shared memory block tiles
  Tensor A_warp_tiles = zipped_divide(A_smem, A_warp_tile_shape);
  Tensor B_warp_tiles = zipped_divide(B_smem, B_warp_tile_shape);

  // create warp tiles for c,d inside of global memory block tiles, since we only read/write
  // once, we dont load them into shared memory
  // the coalesce removes a level of nesting
  Tensor C_warp_tiles = coalesce(zipped_divide(C_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});
  Tensor D_warp_tiles = coalesce(zipped_divide(D_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});

  // now we can access warp tiles which in this kernel correspond to a single mma instruction
  // using the indices ((_,_), (warp_i, warp_j, block_i, block_j))
  Tensor C_warp_tile = C_warp_tiles(make_coord(_,_), make_coord(warp_m, warp_n, block_m, block_n));
  ldmatrix_m16n8_gmem(C_warp_tile.data(), C_register, N * sizeof(half));
  C_register[0] *= beta;
  C_register[1] *= beta;
  C_register[2] *= beta;
  C_register[3] *= beta;

  const unsigned int k_block_tiles = K / BK_dim;
  const unsigned int k_warp_tiles = BK_dim / WK_dim;

  for (unsigned int block_k = 0; block_k < k_block_tiles; block_k++)
  {
    Tensor A_block_tile = A_block_tiles(make_coord(_,_), make_coord(block_m, block_k));
    Tensor B_block_tile = B_block_tiles(make_coord(_,_), make_coord(block_k, block_n));
    tileMemcpy<BM_dim, BK_dim, half>(A_block_tile.data(), A_smem.data().get(), K, BK_dim);
    tileMemcpy<BK_dim, BN_dim, half>(B_block_tile.data(), B_smem.data().get(), N, BN_dim);
    
    // using CuTe's copy algo results in register use exploding to point where kernel wont launch    
    // copy(A_block_tiles(make_coord(_,_), make_coord(block_m, block_k)), A_smem);
    // copy(B_block_tiles(make_coord(_,_), make_coord(block_k, block_n)), B_smem);
    __syncthreads();
    for (unsigned int warp_k = 0; warp_k < k_warp_tiles; warp_k++)
    {
      Tensor A_warp_tile = A_warp_tiles(make_coord(_,_), make_coord(warp_m, warp_k));
      Tensor B_warp_tile = B_warp_tiles(make_coord(_,_), make_coord(warp_k, warp_n));
      ldmatrix_m16n8(A_warp_tile.data().get(), A_register, BK_dim * sizeof(half));
      ldmatrix_n8k8(B_warp_tile.data().get(), B_register, BN_dim * sizeof(half));
      B_register[0] *= alpha;
      B_register[1] *= alpha;
      mma_sync_m16n8k8(C_register, A_register, B_register, C_register);
    }
    __syncthreads();
  }

  Tensor D_warp_tile = D_warp_tiles(make_coord(_,_), make_coord(threadIdx.y, threadIdx.x / 32, blockIdx.y, blockIdx.x));
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
    const unsigned int ThreadsM = WARPS_PER_BLOCK_M;
    const unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_N;

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


