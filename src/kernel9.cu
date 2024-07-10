#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"

// smaller K dimension + fast index calculation
// kernel 5/6 optimizations combined


template <unsigned int mma_tiles_per_warp_m, unsigned int mma_tiles_per_warp_k>
__device__ __forceinline__ void ldmatrix_a(
  const half* src,
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
  const half* src,
  half (&reg)[4][8][2],
  const unsigned int smem_stride
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
unsigned int NUM_THREADS>
__global__ void
kernel_9(half* A,
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

  // for convenience/readability in index calculations
  const unsigned int A_stride = K;
  const unsigned int B_stride = N;
  const unsigned int CD_stride = N;

  // calculate how many bits of shared memory indices are going to be swizzled, and create masks
  constexpr unsigned int SWIZZLE_BITS_B = int_log2(BN_dim / 8);
  constexpr unsigned int SWIZZLE_MASK_B = 0b1110000 << SWIZZLE_BITS_B;

  // loop bounds, constexpr where possible allows for loop unrolling
  constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
  constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
  constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
  const unsigned int num_block_tiles_k = K / BK_dim;
  
  // calculate block/warp indices
  const unsigned int block_m = blockIdx.y;
  const unsigned int block_n = blockIdx.x;
  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;
  
  extern __shared__ half shmem[];
  half* A_block_smem = shmem;
  half* B_block_smem = &shmem[BM_dim * BK_dim];

  // declare register storage
  // ptx instructions expect uint32_t registers, where each uint32_t is 2 halfs packed together  
  uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
  uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
  uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];
  
  // convenience cast to half for register storage
  half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);
  half (&A_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]>(A_register);
  half (&B_register_) [mma_tiles_per_warp_k][mma_tiles_per_warp_n][2] = reinterpret_cast<half(&)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]>(B_register);

  // accumulators start at 0
  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        acc_register_[mma_m][mma_n][0] = 0;
        acc_register_[mma_m][mma_n][1] = 0;
        acc_register_[mma_m][mma_n][2] = 0;
        acc_register_[mma_m][mma_n][3] = 0;
      }
  }

  // these register arrays are used to cache values pre-fetched from global memory during the inner loop of the kernel
  // the code is nicer if we hard code it for these tile dimensions and number of threads
  // since we performing this copy with float4 pointers, for these tile dimensions it works out to be 8 float4s for A and 4 float4s for B
  static_assert(BM_dim == 256);
  static_assert(BN_dim == 128);
  static_assert(BK_dim == 32);
  static_assert(NUM_THREADS == 256);
  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[2];
    
  // prefetch the first block tile of A,B into shared memory
  half* A_block_gmem = A + (block_m * BM_dim * A_stride);
  half* B_block_gmem = B + (block_n * BN_dim);
  tileMemcpySwizzleA<BM_dim, NUM_THREADS>(A_block_gmem, A_block_smem, K);
  tileMemcpySwizzle<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B>(B_block_gmem, B_block_smem, N);

  // construct const pointers to warp tiles for use inside the inner loop
  const half* A_warp_tile = A_block_smem + (warp_m * WM_dim * BK_dim);
  const half* B_warp_tile = B_block_smem + (warp_n * WN_dim);
  const uint32_t A_warp_tile_byte_offset = cvta_to_shared_u32(A_warp_tile);
  const uint32_t B_warp_tile_byte_offset = cvta_to_shared_u32(B_warp_tile);

  for (unsigned int block_k = 1; block_k <= num_block_tiles_k; block_k++)
  {
    __syncthreads();

    if (block_k != num_block_tiles_k)
    {
      half* A_block_gmem = A + (block_m * BM_dim * A_stride) + (block_k * BK_dim);
      half* B_block_gmem = B + (block_k * BK_dim * B_stride) + (block_n * BN_dim);
      tileMemcpyLoad<BM_dim, BK_dim, NUM_THREADS, 4>(A_block_gmem, A_gmem_cache_reg, K);
      tileMemcpyLoad<BK_dim, BN_dim, NUM_THREADS, 2>(B_block_gmem, B_gmem_cache_reg, N);
    }

    ldmatrix_a<mma_tiles_per_warp_m, mma_tiles_per_warp_k>(A_warp_tile, A_register_, BK_dim);

    ldmatrix_b(B_warp_tile, B_register_, BN_dim);

    // outer product between mma tiles
    #pragma unroll
    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++)
    {
      #pragma unroll
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        #pragma unroll
        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
        {
          asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3}, "
            "{%4}, "
            "{%5, %6};"
            : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
            : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
              "r"(B_register[mma_k][mma_n])
              "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
          );
        }
      }
    }
    __syncthreads();

    if (block_k != num_block_tiles_k)
    {
      tileMemcpySwizzleStoreA<BM_dim, NUM_THREADS, 4>(A_gmem_cache_reg, A_block_smem);
      tileMemcpySwizzleStore<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B, 2>(B_gmem_cache_reg, B_block_smem);
    }
  }

  //////////////
  // epilogue //
  //////////////
  half alpha_ = (half)alpha;
  half beta_ = (half)beta;

  half C_register_[128];
  float4 (&C_register) [16] = reinterpret_cast<float4(&)[16]>(C_register_);
  
  // calculate pointers for this warps C and D tiles
  half* C_block_gmem = C + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
  half* C_warp_gmem = C_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);
  half* D_block_gmem = D + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
  half* D_warp_gmem = D_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);
  tileMemcpyLoad<WM_dim, WN_dim, 32, 16>(C_warp_gmem, C_register, N);

  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        // scale C by beta
        unsigned int C_index = mma_m * MMA_M_dim + mma_n * MMA_N_dim;
        acc_register_[mma_m][mma_n][0] = acc_register_[mma_m][mma_n][0] * alpha_ + C_register_[C_index];
        acc_register_[mma_m][mma_n][1] = acc_register_[mma_m][mma_n][1] * alpha_ + C_register_[C_index + 1];
        acc_register_[mma_m][mma_n][2] = acc_register_[mma_m][mma_n][2] * alpha_ + C_register_[C_index + 2];
        acc_register_[mma_m][mma_n][3] = acc_register_[mma_m][mma_n][3] * alpha_ + C_register_[C_index + 3];
      }
  }

  float4 (&acc_register_tmp_)[16] = reinterpret_cast<float4(&)[16]>(acc_register);
  tileMemcpyStore<WM_dim, WN_dim, 32, 16>(acc_register_tmp_, D_warp_gmem, N/8);
}

void kernel_9_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
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
    constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_9<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    65536)); // set shared memory limit to 64KB which is maximum for sm_75

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_9
        <BM_dim, BN_dim, BK_dim,
        WM_dim, WN_dim, WK_dim, NumThreads>
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


