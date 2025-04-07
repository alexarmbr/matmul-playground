
#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <cudaTypedefs.h>

template<unsigned int BM_dim, unsigned int BN_dim>
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) half smem_buffer[BM_dim][BN_dim];
   __shared__ alignas(1024) half smem_buffer_tr[BN_dim][BM_dim];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ cuda::barrier<cuda::thread_scope_block> bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
     cuda::device::experimental::fence_proxy_async_shared_cta();
   }

   __syncthreads();

   cuda::barrier<cuda::thread_scope_block>::arrival_token token;
   if (threadIdx.x == 0) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, 0, 0, bar);
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

//    /* Matrix transpose
//     *  When using the normal shared memory layout, there are eight
//     *  8-way shared memory bank conflict when storing to the transpose.
//     *  When enabling the 128-byte swizzle pattern and using the according access pattern,
//     *  they are eliminated both for load and store. */
//    for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
//       for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
//          const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
//          const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
//          smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
//       }
//    }

//    // Wait for shared memory writes to be visible to TMA engine.
//    cuda::device::experimental::fence_proxy_async_shared_cta();
//    __syncthreads();

//    /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
//     * it will 'unswizzle' the data. */
//    if (threadIdx.x == 0) {
//       cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, 0, 0, &smem_buffer_tr);
//       cuda::device::experimental::cp_async_bulk_commit_group();
//       cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
//    }

//    // Destroy barrier
//    if (threadIdx.x == 0) {
//      (&bar)->~barrier();
//    }
}

// --------------------------------- main ----------------------------------------
using half = __half;
int main(){
    
    
    constexpr unsigned int BM_dim = 128;
    constexpr unsigned int BN_dim = 128;
    constexpr unsigned int shmemNumBytes = (BM_dim * BN_dim) * sizeof(half); 
    
    
    half* d_data;
    size_t data_size = shmemNumBytes;  // Based on dimensions in the tensor setup
    cudaError_t err = cudaMalloc(&d_data, data_size);
    if (err != cudaSuccess) {
        printf("CUDA Memory allocation error: %s\n", cudaGetErrorString(err));
        return -1;
    }


   void* tensor_ptr = d_data;
   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {BN_dim, BM_dim};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {BM_dim * sizeof(half)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {BN_dim, BM_dim};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<BM_dim, BN_dim><<<1, 8>>>(tensor_map);
}