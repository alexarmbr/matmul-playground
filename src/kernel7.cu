#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int NUM_THREADS>
__global__ void
kernel_7(half* A,
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

  // for convenience/readability in index calculations
  const unsigned int A_stride = K;
  const unsigned int B_stride = N;
  const unsigned int CD_stride = N;

  // calculate how many bits of shared memory indices are going to be swizzled, and create masks
  constexpr unsigned int SWIZZLE_BITS_B = int_log2(BN_dim / 8);

  // loop bounds, constexpr where possible allows for loop unrolling
  constexpr unsigned int mma_tiles_per_warp_k = 4;
  constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
  constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
  const unsigned int num_block_tiles_k = K / BK_dim;
  
  // calculate block/warp indices
  const unsigned int block_m = blockIdx.y;
  const unsigned int block_n = blockIdx.x;
  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;
  
  // double buffering
  extern __shared__ half shmem[];
  half* A_block_smem = shmem;
  half* B_block_smem = &shmem[BM_dim * BK_dim];
  // constexpr int BUFFER_SIZE = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(half);
  constexpr int BUFFER_SIZE = (BM_dim * BK_dim + BK_dim * BN_dim);

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
  static_assert(BN_dim == 256);
  static_assert(BK_dim == 32);
  static_assert(NUM_THREADS == 256);
  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[4];
    
  // prefetch the first block tile of A,B into shared memory
  half* A_block_gmem = A + (block_m * BM_dim * A_stride);
  half* B_block_gmem = B + (block_n * BN_dim);
  tileMemcpySwizzleA<BM_dim, NUM_THREADS>(A_block_gmem, A_block_smem, K);
  tileMemcpySwizzle<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B>(B_block_gmem, B_block_smem, N);

  // construct pointers to warp tiles for use inside the inner loop
  half* A_warp_tile = A_block_smem + (warp_m * WM_dim * BK_dim);
  half* B_warp_tile = B_block_smem + (warp_n * WN_dim);

  // calculate pointers into warp tiles
  unsigned int A_logical_offset = (threadIdx.x % 32) * BK_dim;
  unsigned int A_swizzled_offset = A_logical_offset ^ ((A_logical_offset & 0b10000000) >> 4);
  A_swizzled_offset = A_swizzled_offset ^ ((A_swizzled_offset & 0b1100000) >> 2);
  int32_t A_warp_smem = cvta_to_shared_u32(A_warp_tile + A_swizzled_offset);
  constexpr unsigned int A_smem_stride_bytes = BK_dim * sizeof(half);

  const unsigned int B_logical_offset = ((threadIdx.x % 8) * BN_dim) +  (((threadIdx.x % 32) / 8) * 8);
  unsigned int B_swizzled_offset = B_logical_offset ^ ((B_logical_offset & 0b11100000000) >> 5);
  int32_t B_warp_smem = cvta_to_shared_u32(B_warp_tile + B_swizzled_offset);
  constexpr unsigned int B_smem_stride_bytes = BN_dim * sizeof(half);

  int offset_direction = 1;

  for (unsigned int block_k = 1; block_k <= num_block_tiles_k; block_k++)
  {
    __syncthreads();

    A_block_gmem += BK_dim;
    B_block_gmem += BK_dim * B_stride;
    if (block_k != num_block_tiles_k)
    {
      tileMemcpyLoad<BM_dim, BK_dim, NUM_THREADS, 4>(A_block_gmem, A_gmem_cache_reg, K);
      tileMemcpyLoad<BK_dim, BN_dim, NUM_THREADS, 4>(B_block_gmem, B_gmem_cache_reg, N);
    }

    ////////////////////////////
    // load a tile from shmem //
    ////////////////////////////

    // k=0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[0][0][0]), "=r"(A_register[0][0][1]), "=r"(A_register[1][0][0]), "=r"(A_register[1][0][1])
      : "r"(A_warp_smem)
    );

    // k=0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[2][0][0]), "=r"(A_register[2][0][1]), "=r"(A_register[3][0][0]), "=r"(A_register[3][0][1])
      : "r"(A_warp_smem + 32 * A_smem_stride_bytes)
    );

    // k=0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[4][0][0]), "=r"(A_register[4][0][1]), "=r"(A_register[5][0][0]), "=r"(A_register[5][0][1])
      : "r"(A_warp_smem + 64 * A_smem_stride_bytes)
    );

    // k=0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[6][0][0]), "=r"(A_register[6][0][1]), "=r"(A_register[7][0][0]), "=r"(A_register[7][0][1])
      : "r"(A_warp_smem + 96 * A_smem_stride_bytes)
    );

    A_warp_smem ^= 0b10000;
    
    // k=1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[0][1][0]), "=r"(A_register[0][1][1]), "=r"(A_register[1][1][0]), "=r"(A_register[1][1][1])
      : "r"(A_warp_smem)
    );

    // k=1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[2][1][0]), "=r"(A_register[2][1][1]), "=r"(A_register[3][1][0]), "=r"(A_register[3][1][1])
      : "r"(A_warp_smem + 32 * A_smem_stride_bytes)
    );

    // k=1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[4][1][0]), "=r"(A_register[4][1][1]), "=r"(A_register[5][1][0]), "=r"(A_register[5][1][1])
      : "r"(A_warp_smem + 64 * A_smem_stride_bytes)
    );

    // k=1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[6][1][0]), "=r"(A_register[6][1][1]), "=r"(A_register[7][1][0]), "=r"(A_register[7][1][1])
      : "r"(A_warp_smem + 96 * A_smem_stride_bytes)
    );
    
    A_warp_smem ^= 0b110000;

    // k=2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[0][2][0]), "=r"(A_register[0][2][1]), "=r"(A_register[1][2][0]), "=r"(A_register[1][2][1])
      : "r"(A_warp_smem)
    );

    // k=2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[2][2][0]), "=r"(A_register[2][2][1]), "=r"(A_register[3][2][0]), "=r"(A_register[3][2][1])
      : "r"(A_warp_smem + 32 * A_smem_stride_bytes)
    );

    // k=2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[4][2][0]), "=r"(A_register[4][2][1]), "=r"(A_register[5][2][0]), "=r"(A_register[5][2][1])
      : "r"(A_warp_smem + 64 * A_smem_stride_bytes)
    );

    // k=2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[6][2][0]), "=r"(A_register[6][2][1]), "=r"(A_register[7][2][0]), "=r"(A_register[7][2][1])
      : "r"(A_warp_smem + 96 * A_smem_stride_bytes)
    );
    A_warp_smem ^= 0b10000;

    // k=3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[0][3][0]), "=r"(A_register[0][3][1]), "=r"(A_register[1][3][0]), "=r"(A_register[1][3][1])
      : "r"(A_warp_smem)
    );

    // k=3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[2][3][0]), "=r"(A_register[2][3][1]), "=r"(A_register[3][3][0]), "=r"(A_register[3][3][1])
      : "r"(A_warp_smem + 32 * A_smem_stride_bytes)
    );

    // k=3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[4][3][0]), "=r"(A_register[4][3][1]), "=r"(A_register[5][3][0]), "=r"(A_register[5][3][1])
      : "r"(A_warp_smem + 64 * A_smem_stride_bytes)
    );

    // k=3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(A_register[6][3][0]), "=r"(A_register[6][3][1]), "=r"(A_register[7][3][0]), "=r"(A_register[7][3][1])
      : "r"(A_warp_smem + 96 * A_smem_stride_bytes)
    );

    A_warp_smem ^= 0b110000;

    ////////////////////////////
    // load b tile from shmem //
    ////////////////////////////

    int32_t B_warp_smem_ = B_warp_smem;
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[0][0]), "=r"(B_register[0][1]), "=r"(B_register[0][2]), "=r"(B_register[0][3])
      : "r"(B_warp_smem)
    );
  
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[0][4]), "=r"(B_register[0][5]), "=r"(B_register[0][6]), "=r"(B_register[0][7])
      : "r"(B_warp_smem ^ 0b1000000)
    );

    B_warp_smem_ += 8 * B_smem_stride_bytes;

    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[1][0]), "=r"(B_register[1][1]), "=r"(B_register[1][2]), "=r"(B_register[1][3])
      : "r"(B_warp_smem_)
    );
  
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[1][4]), "=r"(B_register[1][5]), "=r"(B_register[1][6]), "=r"(B_register[1][7])
      : "r"(B_warp_smem_ ^ 0b1000000)
    );
  
    B_warp_smem_ += 8 * B_smem_stride_bytes;

    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[2][0]), "=r"(B_register[2][1]), "=r"(B_register[2][2]), "=r"(B_register[2][3])
      : "r"(B_warp_smem_)
    );
  
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[2][4]), "=r"(B_register[2][5]), "=r"(B_register[2][6]), "=r"(B_register[2][7])
      : "r"(B_warp_smem_ ^ 0b1000000)
    );
  
    B_warp_smem_ += 8 * B_smem_stride_bytes;
  
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[3][0]), "=r"(B_register[3][1]), "=r"(B_register[3][2]), "=r"(B_register[3][3])
      : "r"(B_warp_smem_)
    );
  
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(B_register[3][4]), "=r"(B_register[3][5]), "=r"(B_register[3][6]), "=r"(B_register[3][7])
      : "r"(B_warp_smem_ ^ 0b1000000)
    );
    
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

    if (block_k != num_block_tiles_k)
    {
      // switch smem buffers each iteration
      A_warp_smem = A_warp_smem + (BUFFER_SIZE * 2) * offset_direction;
      A_block_smem = A_block_smem + BUFFER_SIZE * offset_direction;
      B_warp_smem = B_warp_smem + (BUFFER_SIZE * 2) * offset_direction;
      B_block_smem = B_block_smem + BUFFER_SIZE * offset_direction;
      offset_direction = -1 * offset_direction;
      tileMemcpySwizzleStoreA<BM_dim, NUM_THREADS, 4>(A_gmem_cache_reg, A_block_smem);
      tileMemcpySwizzleStore<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B, 4>(B_gmem_cache_reg, B_block_smem);
    }
  }

  //////////////
  // epilogue //
  //////////////
  half alpha_ = (half)alpha;
  half beta_ = (half)beta;
  half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
  
  // calculate pointers for this warps C and D tiles
  half* C_block_gmem = C + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
  half* C_warp_gmem = C_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);
  half* D_block_gmem = D + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
  half* D_warp_gmem = D_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);

  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        half* C_mma_tile = C_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
        ldmatrix_m16n8_gmem(C_mma_tile, C_register[mma_m][mma_n], N * sizeof(half));
          
        // scale C by beta
        acc_register_[mma_m][mma_n][0] = acc_register_[mma_m][mma_n][0] * alpha_ + C_register[mma_m][mma_n][0] * beta_;
        acc_register_[mma_m][mma_n][1] = acc_register_[mma_m][mma_n][1] * alpha_ + C_register[mma_m][mma_n][1] * beta_;
        acc_register_[mma_m][mma_n][2] = acc_register_[mma_m][mma_n][2] * alpha_ + C_register[mma_m][mma_n][2] * beta_;
        acc_register_[mma_m][mma_n][3] = acc_register_[mma_m][mma_n][3] * alpha_ + C_register[mma_m][mma_n][3] * beta_;
      }
  }

  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        half* D_mma_tile = D_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
        stmatrix_m16n8(D_mma_tile, acc_register_[mma_m][mma_n], N * sizeof(half));
      }
  }
}

void kernel_7_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    
  constexpr unsigned int BM_dim = 256;
  constexpr unsigned int BN_dim = 256;
  constexpr unsigned int BK_dim = 32;
  
  constexpr unsigned int WARPS_PER_BLOCK_M = 2;
  constexpr unsigned int WARPS_PER_BLOCK_N = 4;
  constexpr unsigned int WARPS_PER_BLOCK_K = 4;
  // WM = 128
  // WN = 64
  // WK = 8

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
    const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * 2 * sizeof(half);

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_7<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    65536)); // set shared memory limit to 64KB which is maximum for sm_75

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_7
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


