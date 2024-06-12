#include <cuda.h>
#include <mma.h>
#include <cute/tensor.hpp>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"
#include "cute_utils.cuh"

using namespace cute;


template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int num_threads>
__global__ void
kernel_10(half* A,
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

  // loop bounds
  constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
  constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
  constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
  constexpr unsigned int warp_tiles_per_block_k = BK_dim / WK_dim;
  const unsigned int block_tiles_k = K / BK_dim;
  
  // const unsigned int blocks_per_M = M / BM_dim;
  const unsigned int blocks_per_N = N / BN_dim;
  // auto swizzle_tile_dim = Int<8>{};
  // const int block_swizzle_tiles_per_M = blocks_per_M / swizzle_tile_dim;
  // const int block_swizzle_tiles_per_N = blocks_per_N / swizzle_tile_dim;
  // Layout block_n_map = make_layout(
  //   make_shape(swizzle_tile_dim, swizzle_tile_dim, block_swizzle_tiles_per_N, block_swizzle_tiles_per_M),
  //   make_stride(1 ,0, swizzle_tile_dim, 0)
  // );

  // Layout block_m_map = make_layout(
  //     make_shape(swizzle_tile_dim, swizzle_tile_dim, block_swizzle_tiles_per_N, block_swizzle_tiles_per_M),
  //     make_stride(0, 1, 0, swizzle_tile_dim)
  // );
  
  // const unsigned int block_m = block_m_map(blockIdx.x);
  // const unsigned int block_n = block_n_map(blockIdx.x);
  const unsigned int block_m = blockIdx.x / blocks_per_N;
  const unsigned int block_n = blockIdx.x % blocks_per_N;
  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;

  auto A_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BK_dim>{});
  auto B_block_tile_shape = make_shape(Int<BK_dim>{}, Int<BN_dim>{});
  auto CD_block_tile_shape = make_shape(Int<BM_dim>{}, Int<BN_dim>{});
  auto A_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WK_dim>{});
  auto B_warp_tile_shape = make_shape(Int<WK_dim>{}, Int<WN_dim>{});
  auto CD_warp_tile_shape = make_shape(Int<WM_dim>{}, Int<WN_dim>{});
  auto A_mma_tile_shape = make_shape(Int<MMA_M_dim>{}, Int<MMA_K_dim>{});
  auto B_mma_tile_shape = make_shape(Int<MMA_K_dim>{}, Int<MMA_N_dim>{});
  auto CD_mma_tile_shape = make_shape(Int<MMA_M_dim>{}, Int<MMA_N_dim>{});

  // double buffering
  extern __shared__ half shmem[];
  half* A_smem_[2] = {shmem, &shmem[BM_dim * BK_dim]};
  half* B_smem_[2] = {&shmem[2 * BM_dim * BK_dim], &shmem[2 * BM_dim * BK_dim + BK_dim * BN_dim]};

  Tensor A_gmem = make_tensor(A, make_shape(M, K), LayoutRight{});
  Tensor B_gmem = make_tensor(B, make_shape(K, N), LayoutRight{});
  Tensor C_gmem = make_tensor(C, make_shape(M, N), LayoutRight{});
  Tensor D_gmem = make_tensor(D, make_shape(M, N), LayoutRight{});

  // block tile each matrix
  Tensor A_block_tiles = zipped_divide(A_gmem, A_block_tile_shape);
  Tensor B_block_tiles = zipped_divide(B_gmem, B_block_tile_shape);
  Tensor C_block_tiles = zipped_divide(C_gmem, CD_block_tile_shape);
  Tensor D_block_tiles = zipped_divide(D_gmem, CD_block_tile_shape);
  
  // create warp tiles for a,b inside of shared memory block tiles
  // Tensor A_warp_tiles = coalesce(zipped_divide(A_smem, A_warp_tile_shape), Step<_1,Step<>>{});
  // Tensor B_warp_tiles = coalesce(zipped_divide(B_smem, B_warp_tile_shape), Step<_1,Step<>>{});
  // Tensor B_warp_tiles = zipped_divide(B_smem, B_warp_tile_shape);
  // if (thread0())
  // {
  //   print(A_warp_tiles.layout());
  // }

  // create mma tiles for a,b inside of warp_tiles
  // Tensor A_mma_tiles = coalesce(zipped_divide(A_warp_tiles, make_shape(A_mma_tile_shape)), Step<_1,Step<>>{});
  // Tensor B_mma_tiles = coalesce(zipped_divide(B_warp_tiles, make_shape(B_mma_tile_shape)), Step<_1,Step<>>{});

  // create warp and mma tiles for c,d inside of global memory block tiles
  Tensor C_warp_tiles = coalesce(zipped_divide(C_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});
  Tensor D_warp_tiles = coalesce(zipped_divide(D_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});
  Tensor C_mma_tiles = coalesce(zipped_divide(C_warp_tiles, make_shape(CD_mma_tile_shape)), Step<_1,_1>{});
  Tensor D_mma_tiles = coalesce(zipped_divide(D_warp_tiles, make_shape(CD_mma_tile_shape)), Step<_1,_1>{});

  // declare register storage for accumulators
  half acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
  
  // A/B accumulators hold two k slices for overlap of data transfer and compute
  // each iteration of the inner loop one slice is being used for compute
  // while the next slice (mod 2) is being written to
  half A_mma_tile_reg[mma_tiles_per_warp_m][2][4];
  half B_mma_tile_reg[2][mma_tiles_per_warp_n][2];
  uint32_t (&A_mma_tile_reg_) [mma_tiles_per_warp_m][2][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][2][2]>(A_mma_tile_reg);
  uint32_t (&B_mma_tile_reg_) [2][mma_tiles_per_warp_n] = reinterpret_cast<uint32_t(&)[2][mma_tiles_per_warp_n]>(B_mma_tile_reg);

  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[2];

  // set up pointers into shared memory tile for A
  const uint32_t A_smem_warp_tile_[2] = {cvta_to_shared_u32(A_smem_[0] + (warp_m * WM_dim) * BK_dim), cvta_to_shared_u32(A_smem_[1] + (warp_m * WM_dim) * BK_dim)};
  uint32_t A_offset_1 = (threadIdx.x % 32) * BK_dim;
  uint32_t A_offset_2 = ((threadIdx.x % 32) + 32) * BK_dim;
  A_offset_1 = A_offset_1 ^ ((A_offset_1 & 0b10000000) >> 4);
  A_offset_2 = A_offset_2 ^ ((A_offset_2 & 0b10000000) >> 4);
  A_offset_1 = A_offset_1 ^ ((A_offset_1 & 0b1100000) >> 2);
  A_offset_2 = A_offset_2 ^ ((A_offset_2 & 0b1100000) >> 2);
  A_offset_1 <<= 1; // convert from half offset to byte offset
  A_offset_2 <<= 1;
  // const int A_increment_xor_patterns[4] = {
  //   0b10000,
  //   0b110000,
  //   0b10000,
  //   0b110000
  // };

  // set up pointers into shared memory tile for B
  const uint32_t B_smem_warp_tile_[2] = {cvta_to_shared_u32(B_smem_[0] + (warp_n * WN_dim)), cvta_to_shared_u32(B_smem_[1] + (warp_n * WN_dim))};
  const unsigned int thread_group = (threadIdx.x % 32) / 8;
  const unsigned int thread_row = threadIdx.x % 8;
  const unsigned int B_logical_offset = (thread_row * BN_dim) + (thread_group * 8);
  unsigned int B_offset_1 = B_logical_offset ^ ((B_logical_offset & 0b1111000000) >> 4);
  unsigned int B_offset_2 = B_offset_1 ^ 0b111000;
  B_offset_1 <<= 1;
  B_offset_2 <<= 1;
  constexpr unsigned int row_offset = BN_dim * 8 * sizeof(half);

  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        acc_register[mma_m][mma_n][0] = 0;
        acc_register[mma_m][mma_n][1] = 0;
        acc_register[mma_m][mma_n][2] = 0;
        acc_register[mma_m][mma_n][3] = 0;
      }
  }

  // copy 0th block tile from gmem -> smem
  Tensor A_block_tile = A_block_tiles(make_coord(_,_), make_coord(block_m, 0));
  Tensor B_block_tile = B_block_tiles(make_coord(_,_), make_coord(0, block_n));
  tileMemcpySwizzleUnrolled_A<BM_dim, BK_dim>(A_block_tile.data(), A_smem_[0], K);
  tileMemcpySwizzleUnrolled_B<BK_dim, BN_dim>(B_block_tile.data(), B_smem_[0], N);
  __syncthreads();

  // copy 0th k slice of A from smem -> register
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(A_mma_tile_reg_[0][0][0]), "=r"(A_mma_tile_reg_[0][0][1]), "=r"(A_mma_tile_reg_[1][0][0]), "=r"(A_mma_tile_reg_[1][0][1])
    : "r"(A_smem_warp_tile_[0] + A_offset_1)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(A_mma_tile_reg_[2][0][0]), "=r"(A_mma_tile_reg_[2][0][1]), "=r"(A_mma_tile_reg_[3][0][0]), "=r"(A_mma_tile_reg_[3][0][1])
    : "r"(A_smem_warp_tile_[0] + A_offset_2)
  );

  // advance offsets
  // A_offset_1 ^= A_increment_xor_patterns[0];
  // A_offset_2 ^= A_increment_xor_patterns[1];
  A_offset_1 ^= 0b10000;
  A_offset_2 ^= 0b10000;


  // copy 0th k slice of B from smem -> register
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(B_mma_tile_reg_[0][0]), "=r"(B_mma_tile_reg_[0][1]), "=r"(B_mma_tile_reg_[0][2]), "=r"(B_mma_tile_reg_[0][3])
    : "r"(B_smem_warp_tile_[0] + B_offset_1)
  );
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(B_mma_tile_reg_[0][7]), "=r"(B_mma_tile_reg_[0][6]), "=r"(B_mma_tile_reg_[0][5]), "=r"(B_mma_tile_reg_[0][4])
    : "r"(B_smem_warp_tile_[0] + B_offset_2)
  );

  // static_assert(BM_dim == 256, "BM_dim must be 256");
  unsigned int smem_buffer_ind = 0;
  for (unsigned int block_k = 1; block_k <= block_tiles_k; block_k++)
  {
    #pragma unroll
    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++)
    {
      if (mma_k == mma_tiles_per_warp_k - 1 && block_k != block_tiles_k)
      {

        // copy from register cache -> smem
        {
          float4* A_smem_float4 = reinterpret_cast<float4*>(A_smem_[block_k % 2]);
          int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
          constexpr unsigned int iterations = BM_dim * (BK_dim / 8) / num_threads;
          
          #pragma unroll
          for (int i = 0; i < iterations; i++)
          {
            unsigned int dst_ind = thread_idx ^ ((thread_idx & 0b10000) >> 4);
            dst_ind = dst_ind ^ ((dst_ind & 0b1100) >> 2);
            A_smem_float4[dst_ind] = A_gmem_cache_reg[i];
            thread_idx += num_threads;
          }
        }
    
        {
          float4* B_smem_float4 = reinterpret_cast<float4*>(B_smem_[block_k % 2]);
          int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
          constexpr unsigned int iterations = BK_dim * (BN_dim / 8) / num_threads;
    
          #pragma unroll
          for (int i = 0; i < iterations; i++)
          {
            const unsigned int dst_ind = thread_idx ^ ((thread_idx & 0b1110000) >> 4);
            B_smem_float4[dst_ind] = B_gmem_cache_reg[i];
            thread_idx += num_threads;
          }
        }
        __syncthreads();
      }

      // copy next k slice of B from smem -> register
      const unsigned int mma_row = (mma_k + 1) % 4;
      if (mma_row == 0)
      {
        smem_buffer_ind = 1 - smem_buffer_ind;
      }

      // load next k slice of A from smem -> register
      const unsigned int register_load_iter  = (mma_k + 1) % 2;
      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(A_mma_tile_reg_[0][register_load_iter][0]), "=r"(A_mma_tile_reg_[0][register_load_iter][1]), "=r"(A_mma_tile_reg_[1][register_load_iter][0]), "=r"(A_mma_tile_reg_[1][register_load_iter][1])
        : "r"(A_smem_warp_tile_[smem_buffer_ind] + A_offset_1)
      );
    
      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(A_mma_tile_reg_[2][register_load_iter][0]), "=r"(A_mma_tile_reg_[2][register_load_iter][1]), "=r"(A_mma_tile_reg_[3][register_load_iter][0]), "=r"(A_mma_tile_reg_[3][register_load_iter][1])
        : "r"(A_smem_warp_tile_[smem_buffer_ind] + A_offset_2)
      );
      // A_offset_1 ^= A_increment_xor_patterns[(mma_k + 1) % 4];
      // A_offset_2 ^= A_increment_xor_patterns[(mma_k + 1) % 4];
      if (mma_k % 2 == 0)
      {
        A_offset_1 ^= 0b110000;
        A_offset_2 ^= 0b110000;
      }
      else
      {
        A_offset_1 ^= 0b10000;
        A_offset_2 ^= 0b10000;
      }


      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(B_mma_tile_reg_[register_load_iter][0]), "=r"(B_mma_tile_reg_[register_load_iter][1]), "=r"(B_mma_tile_reg_[register_load_iter][2]), "=r"(B_mma_tile_reg_[register_load_iter][3])
        : "r"(B_smem_warp_tile_[smem_buffer_ind] + (B_offset_1 + (mma_row * row_offset)))
      );
      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(B_mma_tile_reg_[register_load_iter][7]), "=r"(B_mma_tile_reg_[register_load_iter][6]), "=r"(B_mma_tile_reg_[register_load_iter][5]), "=r"(B_mma_tile_reg_[register_load_iter][4])
        : "r"(B_smem_warp_tile_[smem_buffer_ind] + (B_offset_2 + (mma_row * row_offset)))
      );

    // if (thread0())
    // {
    //   // printf("%d: current: %f, prev: %f\n", mma_row, (float) B_mma_tile_reg[register_load_iter][0][0], (float) B_mma_tile_reg[1-register_load_iter][0][0]);
    //   printf("%d: current: %f, prev: %f\n", mma_row, (float) A_mma_tile_reg[0][register_load_iter][0], (float) B_mma_tile_reg[0][1-register_load_iter][0]);
    // }
    if (mma_k == 0 && block_k != block_tiles_k)
    {
        Tensor A_block_tile = A_block_tiles(make_coord(_,_), make_coord(block_m, block_k));
        Tensor B_block_tile = B_block_tiles(make_coord(_,_), make_coord(block_k, block_n));
        // copy tile of A from global memory to registers
        // we want these memory requests to be in flight while the mmas are being computed
        {
          constexpr unsigned int float4_cols = BK_dim / 8; // 8
          Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(A_block_tile.data()), make_shape(BM_dim, float4_cols), make_stride(K / 8, 1));
          unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
          const unsigned int thread_idx_y = thread_idx / float4_cols;
          const unsigned int thread_idx_x = thread_idx % float4_cols;
  
          A_gmem_cache_reg[0] = src_float4(thread_idx_y, thread_idx_x);
          A_gmem_cache_reg[1] = src_float4(thread_idx_y + 32, thread_idx_x);
          A_gmem_cache_reg[2] = src_float4(thread_idx_y + 64, thread_idx_x);
          A_gmem_cache_reg[3] = src_float4(thread_idx_y + 96, thread_idx_x);
          // A_gmem_cache_reg[4] = src_float4(thread_idx_y + 128, thread_idx_x);
          // A_gmem_cache_reg[5] = src_float4(thread_idx_y + 160, thread_idx_x);
          // A_gmem_cache_reg[6] = src_float4(thread_idx_y + 192, thread_idx_x);
          // A_gmem_cache_reg[7] = src_float4(thread_idx_y + 224, thread_idx_x);
        }
  
        // copy tile of B from global memory to registers
        {
          constexpr unsigned int float4_cols = BN_dim / 8; // 16
          Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(B_block_tile.data()), make_shape(BK_dim, float4_cols), make_stride(N / 8, 1));
          unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
          const unsigned int thread_idx_y = thread_idx / float4_cols;
          const unsigned int thread_idx_x = thread_idx % float4_cols;
          B_gmem_cache_reg[0] = src_float4(thread_idx_y, thread_idx_x);
          B_gmem_cache_reg[1] = src_float4(thread_idx_y + 16, thread_idx_x);
          // B_gmem_cache_reg[2] = src_float4(thread_idx_y + 32, thread_idx_x);
          // B_gmem_cache_reg[3] = src_float4(thread_idx_y + 48, thread_idx_x);
        }
    }


    const unsigned int register_compute_iter = 1 - register_load_iter;
    #pragma unroll
    for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
    {
      #pragma unroll
      for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
      {
        mma_sync_m16n8k8(
          acc_register[mma_m][mma_n],
          A_mma_tile_reg[mma_m][register_compute_iter],
          B_mma_tile_reg[register_compute_iter][mma_n],
          acc_register[mma_m][mma_n]
        );
      }
    }
  }

  }



  half alpha_ = (half)alpha;
  half beta_ = (half)beta;
  half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        Tensor C_mma_tile = C_mma_tiles(make_coord(_,_), make_coord(mma_m, mma_n, warp_m, warp_n, block_m, block_n));
        ldmatrix_m16n8_gmem(C_mma_tile.data(), C_register[mma_m][mma_n], N * sizeof(half));
        acc_register[mma_m][mma_n][0] = acc_register[mma_m][mma_n][0] * alpha_ + C_register[mma_m][mma_n][0] * beta_;
        acc_register[mma_m][mma_n][1] = acc_register[mma_m][mma_n][1] * alpha_ + C_register[mma_m][mma_n][1] * beta_;
        acc_register[mma_m][mma_n][2] = acc_register[mma_m][mma_n][2] * alpha_ + C_register[mma_m][mma_n][2] * beta_;
        acc_register[mma_m][mma_n][3] = acc_register[mma_m][mma_n][3] * alpha_ + C_register[mma_m][mma_n][3] * beta_;
      }
  }

  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        Tensor D_mma_tile = D_mma_tiles(make_coord(_,_), make_coord(mma_m, mma_n, warp_m, warp_n, block_m, block_n));
        stmatrix_m16n8(D_mma_tile.data(), acc_register[mma_m][mma_n], N * sizeof(half));
      }
  }
}

void kernel_10_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    
  constexpr unsigned int BM_dim = 256;
  constexpr unsigned int BN_dim = 128;
  constexpr unsigned int BK_dim = 32;
  
  constexpr unsigned int WARPS_PER_BLOCK_M = 4;
  constexpr unsigned int WARPS_PER_BLOCK_N = 2;
  constexpr unsigned int WARPS_PER_BLOCK_K = 1;

    constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
    constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
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
    constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
    constexpr unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_N;
    constexpr unsigned int num_threads = ThreadsM * ThreadsN;
    constexpr unsigned int shmem_bytes = 2 * (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);
    // constexpr unsigned int A_swizzle_bits = int_log2(BK_dim/8);
    // constexpr unsigned int B_swizzle_bits = int_log2(BN_dim/8);

    dim3 gridDim(BlocksN * BlocksM, 1);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_10<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, num_threads>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    65536)); // set shared memory limit to 64KB which is maximum for sm_75

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_10
        <BM_dim, BN_dim, BK_dim,
        WM_dim, WN_dim, WK_dim, num_threads>
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


