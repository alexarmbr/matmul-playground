#include <cuda.h>
#include <mma.h>
#include <cute/tensor.hpp>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"
#include "cute_utils.cuh"

using namespace cute;


template <unsigned int mma_tiles_per_warp_m, unsigned int mma_tiles_per_warp_k>
__device__ __forceinline__ void ldmatrix_a(
  half* src,
  half (&reg)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4],
  const unsigned int smem_stride
)
{
  static_assert(mma_tiles_per_warp_m == 4, "mma_tiles_per_warp_m must be 4");
  static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");

  uint32_t (&reg_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2]>(reg);
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
  swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
    
    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0][0]), "=r"(reg_[0][0][1]), "=r"(reg_[1][0][0]), "=r"(reg_[1][0][1])
      : "r"(src_addr)
    );
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][1][0]), "=r"(reg_[0][1][1]), "=r"(reg_[1][1][0]), "=r"(reg_[1][1][1])
        : "r"(src_addr)
    );
    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][2][0]), "=r"(reg_[0][2][1]), "=r"(reg_[1][2][0]), "=r"(reg_[1][2][1])
        : "r"(src_addr)
    );
    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][3][0]), "=r"(reg_[0][3][1]), "=r"(reg_[1][3][0]), "=r"(reg_[1][3][1])
        : "r"(src_addr)
    );
    src_addr ^= 0b100000110000;


    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][0][0]), "=r"(reg_[2][0][1]), "=r"(reg_[3][0][0]), "=r"(reg_[3][0][1])
      : "r"(src_addr)
    );
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][1][0]), "=r"(reg_[2][1][1]), "=r"(reg_[3][1][0]), "=r"(reg_[3][1][1])
        : "r"(src_addr)
    );
    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][2][0]), "=r"(reg_[2][2][1]), "=r"(reg_[3][2][0]), "=r"(reg_[3][2][1])
        : "r"(src_addr)
    );
    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][3][0]), "=r"(reg_[2][3][1]), "=r"(reg_[3][3][0]), "=r"(reg_[3][3][1])
        : "r"(src_addr)
    );
}


__device__ __forceinline__ void ldmatrix_b(
  half* src,
  half (&reg)[4][8][2],
  const unsigned int smem_stride,
  half alpha

)
{
  uint32_t (&reg_) [4][8] = reinterpret_cast<uint32_t(&)[4][8]>(reg);
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b1111000000) >> 4);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
  // when looking at this addr in debugger, it appears that it is just the number of bytes from the start of the shared memory

  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0]), "=r"(reg_[1][0]), "=r"(reg_[2][0]), "=r"(reg_[3][0])
      : "r"(src_addr)
  );
  src_addr ^= 0b10000;
  
  // 1
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][1]), "=r"(reg_[1][1]), "=r"(reg_[2][1]), "=r"(reg_[3][1])
      : "r"(src_addr)
  );
  src_addr ^= 0b110000;

  // 2
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][2]), "=r"(reg_[1][2]), "=r"(reg_[2][2]), "=r"(reg_[3][2])
      : "r"(src_addr)
  );
  src_addr ^= 0b10000;

  // 3
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][3]), "=r"(reg_[1][3]), "=r"(reg_[2][3]), "=r"(reg_[3][3])
      : "r"(src_addr)
  );
  src_addr ^= 0b1110000;

  // 4
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][4]), "=r"(reg_[1][4]), "=r"(reg_[2][4]), "=r"(reg_[3][4])
      : "r"(src_addr)
  );
  src_addr ^= 0b10000;

  // 5
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][5]), "=r"(reg_[1][5]), "=r"(reg_[2][5]), "=r"(reg_[3][5])
      : "r"(src_addr)
  );
  src_addr ^= 0b110000;
  
  // 6
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][6]), "=r"(reg_[1][6]), "=r"(reg_[2][6]), "=r"(reg_[3][6])
      : "r"(src_addr)
  );
  src_addr ^= 0b10000;

  // 7
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][7]), "=r"(reg_[1][7]), "=r"(reg_[2][7]), "=r"(reg_[3][7])
      : "r"(src_addr)
  );
}


template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int num_threads>
__global__ void
kernel_6(half* A,
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
  const unsigned int num_block_tiles_k = K / BK_dim;
  
  const unsigned int blocks_per_N = N / BN_dim;
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

  extern __shared__ half shmem[];
  half* A_smem_ = shmem;
  half* B_smem_ = &shmem[BM_dim * BK_dim];

  Tensor A_gmem = make_tensor(A, make_shape(M, K), LayoutRight{});
  Tensor B_gmem = make_tensor(B, make_shape(K, N), LayoutRight{});
  Tensor C_gmem = make_tensor(C, make_shape(M, N), LayoutRight{});
  Tensor D_gmem = make_tensor(D, make_shape(M, N), LayoutRight{});

  // block tile each matrix
  Tensor A_block_tiles = zipped_divide(A_gmem, A_block_tile_shape);
  Tensor B_block_tiles = zipped_divide(B_gmem, B_block_tile_shape);
  Tensor C_block_tiles = zipped_divide(C_gmem, CD_block_tile_shape);
  Tensor D_block_tiles = zipped_divide(D_gmem, CD_block_tile_shape);

  // create warp and mma tiles for c,d inside of global memory block tiles
  Tensor C_warp_tiles = coalesce(zipped_divide(C_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});
  Tensor D_warp_tiles = coalesce(zipped_divide(D_block_tiles, make_shape(CD_warp_tile_shape)), Step<_1,_1>{});
  Tensor C_mma_tiles = coalesce(zipped_divide(C_warp_tiles, make_shape(CD_mma_tile_shape)), Step<_1,_1>{});
  Tensor D_mma_tiles = coalesce(zipped_divide(D_warp_tiles, make_shape(CD_mma_tile_shape)), Step<_1,_1>{});

  // declare register storage for accumulators
  half acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
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

  // set up pointers into shared memory tile for A
  half* A_smem_warp = A_smem_ + (warp_m * WM_dim) * BK_dim;
  uint32_t A_offset_1 = (threadIdx.x % 32) * BK_dim;
  uint32_t A_offset_2 = ((threadIdx.x % 32) + 32) * BK_dim;
  A_offset_1 = cvta_to_shared_u32(A_smem_warp + A_offset_1);
  A_offset_2 = cvta_to_shared_u32(A_smem_warp + A_offset_2);
  A_offset_1 = A_offset_1 ^ ((A_offset_1 & 0b100000000) >> 4);
  A_offset_2 = A_offset_2 ^ ((A_offset_2 & 0b100000000) >> 4);
  A_offset_1 = A_offset_1 ^ ((A_offset_1 & 0b11000000) >> 2);
  A_offset_2 = A_offset_2 ^ ((A_offset_2 & 0b11000000) >> 2);

  // if (thread0())
  // {
  //   printf("offset 1: %d, offset 2: %d\n", A_offset_1, A_offset_2);
  // }

  // A_offset_1 <<= 1; // convert from half offset to byte offset
  // A_offset_2 <<= 1;
  // const int A_increment_xor_patterns[4] = {
  //   0b10000,
  //   0b110000,
  //   0b10000,
  //   0b110000
  // };


  Tensor A_block_tile = A_block_tiles(make_coord(_,_), make_coord(block_m, 0));
  Tensor B_block_tile = B_block_tiles(make_coord(_,_), make_coord(0, block_n));
  tileMemcpySwizzleUnrolled_A<BM_dim, BK_dim>(A_block_tile.data(), A_smem_, K);
  tileMemcpySwizzleUnrolled_B<BK_dim, BN_dim>(B_block_tile.data(), B_smem_, N);
  __syncthreads();

  half A_mma_tile_reg[mma_tiles_per_warp_m][4];
  half B_mma_tile_reg[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2];
  uint32_t (&A_mma_tile_reg_) [mma_tiles_per_warp_m][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][2]>(A_mma_tile_reg);
  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[2];
  for (unsigned int block_k = 1; block_k <= num_block_tiles_k; block_k++)
  {
    if (block_k != num_block_tiles_k)
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
      }
    }
 
    // ldmatrix_a
    // <mma_tiles_per_warp_m, mma_tiles_per_warp_k>
    // (
    //   A_smem_ + (warp_m * WM_dim) * BK_dim,
    //   A_mma_tile_reg,
    //   BK_dim
    // );
    ldmatrix_b(
      B_smem_ + (warp_n * WN_dim),
      B_mma_tile_reg,
      BN_dim,
      alpha
    );


    // outer product between tiles of a and b
    #pragma unroll
    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++)
    {
      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(A_mma_tile_reg_[0][0]), "=r"(A_mma_tile_reg_[0][1]), "=r"(A_mma_tile_reg_[1][0]), "=r"(A_mma_tile_reg_[1][1])
        : "r"(A_offset_1)
      );
    
      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(A_mma_tile_reg_[2][0]), "=r"(A_mma_tile_reg_[2][1]), "=r"(A_mma_tile_reg_[3][0]), "=r"(A_mma_tile_reg_[3][1])
        : "r"(A_offset_2)
      );

      #pragma unroll
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        #pragma unroll
        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
        {
          mma_sync_m16n8k8(
            acc_register[mma_m][mma_n],
            A_mma_tile_reg[mma_m],
            B_mma_tile_reg[mma_k][mma_n],
            acc_register[mma_m][mma_n]
          );
        }
      }

      switch (mma_k) {
        case 0:
          A_offset_1 ^= 0b10000;
          A_offset_2 ^= 0b10000;
          break;
        case 1:
          A_offset_1 ^= 0b110000;
          A_offset_2 ^= 0b110000;
          break;
        case 2:
          A_offset_1 ^= 0b10000;
          A_offset_2 ^= 0b10000;
          break;
        case 3:
          A_offset_1 ^= 0b110000;
          A_offset_2 ^= 0b110000;
          break;
      }
    }
    __syncthreads();

    {
      float4* A_smem_float4 = reinterpret_cast<float4*>(A_smem_);
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
      float4* B_smem_float4 = reinterpret_cast<float4*>(B_smem_);
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

void kernel_6_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
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
    constexpr unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);
    // constexpr unsigned int A_swizzle_bits = int_log2(BK_dim/8);
    // constexpr unsigned int B_swizzle_bits = int_log2(BN_dim/8);

    dim3 gridDim(BlocksN * BlocksM, 1);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_6<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, num_threads>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    65536)); // set shared memory limit to 64KB which is maximum for sm_75

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_6
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


