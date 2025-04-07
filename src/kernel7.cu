#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <cudaTypedefs.h>

#include "structs_n_stuff.cuh"

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint32_t x)
{
  return (x & 0x3FFFF) >> 4;
}
  

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
  }

// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
template <unsigned int smem_width_bytes>
__device__ uint64_t make_smem_descriptor(half* smem_ptr)
{
  uint64_t smem_desc = 0;
  smem_desc |= matrix_descriptor_encode(cvta_to_shared_u32(smem_ptr));
  smem_desc |= matrix_descriptor_encode(smem_width_bytes * 8) << 32; // offset in bytes between groups of 8 rows
  smem_desc |= uint64_t(1) << 62; // 128B swizzle
  return smem_desc;
}

__device__ __forceinline__ void wgmma_m64n256k16_f32_f16_f16(float D[128], uint64_t A_desc, uint64_t B_desc){
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, " // D (accumulator registers)
        
        "%128, %129, " // A_desc, B_desc (shared memory descriptors)

        "1, 1, 1, 0, 0;" //  scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b
        // scale-d=1: compute D = A * B + D rather than D = A * B
        // imm-scale-a=1: compute A = A * 1.0f (no scaling, A can optionally be negated if you pass -1)
        // imm-scale-b=1: compute B = B * 1.0f (no scaling, B can optionally be negated if you pass -1)
        // imm-trans-a=0: do not transpose A
        // imm-trans-b=0: do not transpose B
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7]), "=f"(D[8]), "=f"(D[9]), "=f"(D[10]), "=f"(D[11]), "=f"(D[12]), "=f"(D[13]), "=f"(D[14]), "=f"(D[15]),
            "=f"(D[16]), "=f"(D[17]), "=f"(D[18]), "=f"(D[19]), "=f"(D[20]), "=f"(D[21]), "=f"(D[22]), "=f"(D[23]), "=f"(D[24]), "=f"(D[25]), "=f"(D[26]), "=f"(D[27]), "=f"(D[28]), "=f"(D[29]), "=f"(D[30]), "=f"(D[31]),
            "=f"(D[32]), "=f"(D[33]), "=f"(D[34]), "=f"(D[35]), "=f"(D[36]), "=f"(D[37]), "=f"(D[38]), "=f"(D[39]), "=f"(D[40]), "=f"(D[41]), "=f"(D[42]), "=f"(D[43]), "=f"(D[44]), "=f"(D[45]), "=f"(D[46]), "=f"(D[47]),
            "=f"(D[48]), "=f"(D[49]), "=f"(D[50]), "=f"(D[51]), "=f"(D[52]), "=f"(D[53]), "=f"(D[54]), "=f"(D[55]), "=f"(D[56]), "=f"(D[57]), "=f"(D[58]), "=f"(D[59]), "=f"(D[60]), "=f"(D[61]), "=f"(D[62]), "=f"(D[63]),
            "=f"(D[64]), "=f"(D[65]), "=f"(D[66]), "=f"(D[67]), "=f"(D[68]), "=f"(D[69]), "=f"(D[70]), "=f"(D[71]), "=f"(D[72]), "=f"(D[73]), "=f"(D[74]), "=f"(D[75]), "=f"(D[76]), "=f"(D[77]), "=f"(D[78]), "=f"(D[79]),
            "=f"(D[80]), "=f"(D[81]), "=f"(D[82]), "=f"(D[83]), "=f"(D[84]), "=f"(D[85]), "=f"(D[86]), "=f"(D[87]), "=f"(D[88]), "=f"(D[89]), "=f"(D[90]), "=f"(D[91]), "=f"(D[92]), "=f"(D[93]), "=f"(D[94]), "=f"(D[95]),
            "=f"(D[96]), "=f"(D[97]), "=f"(D[98]), "=f"(D[99]), "=f"(D[100]), "=f"(D[101]), "=f"(D[102]), "=f"(D[103]), "=f"(D[104]), "=f"(D[105]), "=f"(D[106]), "=f"(D[107]), "=f"(D[108]), "=f"(D[109]), "=f"(D[110]), "=f"(D[111]),
            "=f"(D[112]), "=f"(D[113]), "=f"(D[114]), "=f"(D[115]), "=f"(D[116]), "=f"(D[117]), "=f"(D[118]), "=f"(D[119]), "=f"(D[120]), "=f"(D[121]), "=f"(D[122]), "=f"(D[123]), "=f"(D[124]), "=f"(D[125]), "=f"(D[126]), "=f"(D[127])
            : "l"(A_desc), "l"(B_desc)
    );
}

template <unsigned int SMEM_HEIGHT, unsigned int SMEM_WIDTH>
void __createTensorMapHost(half* tensor_ptr, unsigned int gmem_height, unsigned int gmem_width, CUtensorMap* tensor_map)
{
  constexpr uint32_t rank = 2;
  uint64_t gmem_size[2] = {uint64_t(gmem_width), uint64_t(gmem_height)};
  uint64_t gmem_stride[1] = {sizeof(half) * gmem_width};
  uint32_t smem_size[2] = {uint32_t(SMEM_WIDTH), uint32_t(SMEM_HEIGHT)};
  uint32_t smem_stride[2] = {1, 1};

  // __nv_bfloat16* bf16_tensor_ptr = reinterpret_cast<__nv_bfloat16*>(tensor_ptr);
  
  // auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    gmem_size,                       // const cuuint64_t *globalDim,
    gmem_stride,                     // const cuuint64_t *globalStrides,
    smem_size,                   // const cuuint32_t *boxDim,
    smem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  assert(res == CUDA_SUCCESS);
}



template <unsigned int SMEM_HEIGHT, unsigned int SMEM_WIDTH>
CUtensorMap* createTensorMap(half* tensor_ptr, unsigned int gmem_height, unsigned int gmem_width)
{
  CUtensorMap tensor_map_host;
  __createTensorMapHost<SMEM_HEIGHT, SMEM_WIDTH>(tensor_ptr, gmem_height, gmem_width, &tensor_map_host);
  CUtensorMap* tensor_map_device;
  CUDA_CHECK(cudaMalloc(&tensor_map_device, sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(tensor_map_device, &tensor_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  return tensor_map_device;
}


template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim>
__global__ void
kernel_7(
  // const CUtensorMap* tensorMapA,
  // const CUtensorMap* tensorMapB,
  const __grid_constant__ CUtensorMap tensorMapA,
  const __grid_constant__ CUtensorMap tensorMapB,
  half* C,
  half* D,
  const float alpha,
  const float beta, 
  const unsigned int M,
  const unsigned int N,
  unsigned int K)
{

  __shared__ alignas(128) half A_block_smem[BM_dim*BK_dim];
  __shared__ alignas(128) half B_block_smem[BK_dim*BN_dim];
  constexpr unsigned int ABlockNumBytes = BM_dim * BK_dim * sizeof(half);
  constexpr unsigned int BBlockNumBytes = BK_dim * BN_dim * sizeof(half);


  // if (threadIdx.x == 0){
  //   // print the first 16 elements of A_block_smem
  //   for (int i = 0; i < 16; i++){
  //     printf("%f ", A_block_smem[i]);
  //   }
  //   printf("\n");
  // }
  
  // this pragma suppresses the warning about static variables with dynamic initialization
  #pragma nv_diag_suppress static_var_with_dynamic_init

  // cuda::thread_scope_block sets the scope of the barrier to the block
  // only one thread will initialize the barrier, but any thread in the block can wait on it
  __shared__ cuda::barrier<cuda::thread_scope_block> barA;
  __shared__ cuda::barrier<cuda::thread_scope_block> barB;

  if (threadIdx.x == 0) {
    
    
    // initialize the barrier, since the entire thread block will wait on it,
    // the arrivall count argument is blockDim.x
    printf("arrival count is %d\n", blockDim.x);
    init(&barA, blockDim.x);
    init(&barB, blockDim.x);

    // this synchronizes with the tensor memory accelerator (TMA)
    // the TMA is a hardware unit that operates asychronously with respect to other stuff happening on the SM
    cuda::device::experimental::fence_proxy_async_shared_cta();
    printf("fence done\n");
  }

  // sychronize so that the initialized barrier is visible to all threads
  __syncthreads();

  cuda::barrier<cuda::thread_scope_block>::arrival_token tokenA, tokenB;
  if (threadIdx.x == 0){
    // printf("copying B\n");
    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&A_block_smem, &tensorMapA, 0, 0, barA);
    tokenA = cuda::device::barrier_arrive_tx(barA, 1, ABlockNumBytes);

    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&B_block_smem, &tensorMapB, 0, 0, barB);
    tokenB = cuda::device::barrier_arrive_tx(barB, 1, BBlockNumBytes);
    // printf("copying B done\n");
    // 0th thread arrives on the barrier
    // barriers have an arrival count (decrement by 1 here)
    // and a transaction count (how many bytes are expected to arrive)
    // printf("token is %d\n", token);
  }
  else {
    tokenA = barA.arrive();
    tokenB = barB.arrive();
  }

  barA.wait(std::move(tokenA));
  barB.wait(std::move(tokenB));

  __syncthreads();


  uint64_t A_smem_desc = make_smem_descriptor<BK_dim * sizeof(half)>(A_block_smem);
  uint64_t B_smem_desc = make_smem_descriptor<BK_dim * sizeof(half)>(B_block_smem);

  float D_reg[128];
  
  wgmma_m64n256k16_f32_f16_f16(D_reg, A_smem_desc, B_smem_desc);

  if (threadIdx.x == 0){
    printf("D_reg is %f %f\n", (float) D_reg[0], (float) D_reg[1]);
  }


  #define OUT_IDX(i, j) i * N + j

  int thread = threadIdx.x % 128;
  int row = thread / 4;
  int col = thread % 4;
  float* out = reinterpret_cast<float*>(D);

  out[OUT_IDX(row, col)] = D_reg[0];
  out[OUT_IDX(row, col)] = D_reg[1];

}
    


void kernel_7_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;
    half* A_ptr = device_sgemm_params.A;
    half* B_ptr = device_sgemm_params.B;
    half* C_ptr = device_sgemm_params.C;
    half* D_ptr = device_sgemm_params.D;

    constexpr unsigned int BM_dim = 64;
    constexpr unsigned int BN_dim = 256;
    constexpr unsigned int BK_dim = 16;
    constexpr unsigned int shmemNumBytes = BM_dim * BK_dim + BK_dim * BN_dim; 


    // CUtensorMap* tensorMapA = createTensorMap<BM_dim, BK_dim>(A_ptr, M, K);
    // CUtensorMap* tensorMapB = createTensorMap<BN_dim, BK_dim>(B_ptr, N, K);
    CUtensorMap tensor_map_A{};
    __createTensorMapHost<BM_dim, BK_dim>(A_ptr, M, K, &tensor_map_A);

    CUtensorMap tensor_map_B{};
    __createTensorMapHost<BN_dim, BK_dim>(B_ptr, N, K, &tensor_map_B);

    // need to set this
    CUDA_CHECK(cudaFuncSetAttribute(kernel_7<BM_dim, BN_dim, BK_dim>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    shmemNumBytes * 2 * sizeof(half)));

    // dim3 gridDimension(device_sgemm_params.M / BM_dim, device_sgemm_params.N / BN_dim);
    dim3 gridDimension(1);
    dim3 blockDimension(128);

    for (int i = 0; i < num_runs; i++)
    {
        timer.Start();
        kernel_7
        <BM_dim, BN_dim, BK_dim>
        <<<gridDimension, blockDimension>>>(
            tensor_map_A,
            tensor_map_B,
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



    half* D_host = new half[M * N];
    CUDA_CHECK(cudaMemcpy(D_host, D_ptr, M * N * sizeof(half), cudaMemcpyDeviceToHost));


    // print top left 16x16 tile of D
    for (int i = 0; i < 16; i++){
      for (int j = 0; j < 16; j++){
        printf("%f ", (float) D_host[i * N + j]);
      }
      printf("\n");
    }
    
}


