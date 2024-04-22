#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <iostream>
#include <array>
#include <vector>
#include "structs_n_stuff.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


template <unsigned int SHMEM_SIZE_BYTES>
__global__ void increment(float* src, float* dst, unsigned int num_elements)
{
    extern __shared__ float shmem[];
    constexpr unsigned int SHMEM_SIZE = SHMEM_SIZE_BYTES / sizeof(float);
    for (unsigned int i = threadIdx.x; i < num_elements; i += SHMEM_SIZE)
    {
        for (unsigned int j = 0; j < SHMEM_SIZE; j+=blockDim.x)
        {
            shmem[j] = src[i+j];
            shmem[j] += 1;
        }

        for (unsigned int j = 0; j < SHMEM_SIZE; j+=blockDim.x)
        {
            dst[i+j] = src[j];
        }
    }
}

template <unsigned int SHMEM_SIZE_BYTES>
__global__ void increment_async(float* src, float* dst, unsigned int num_elements)
{
    extern __shared__ float shmem[];
    constexpr unsigned int PIPELINE_NUM_STAGES = 2;
    constexpr unsigned int SHMEM_SIZE = SHMEM_SIZE_BYTES / sizeof(float);
    constexpr unsigned int SHMEM_CHUNK_SIZE = SHMEM_SIZE / PIPELINE_NUM_STAGES;
    size_t shmem_offset[2] = {0, SHMEM_CHUNK_SIZE};

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    
    // threads in the last warp are producers, other threads are consumers
    const cuda::pipeline_role thread_role =
     block.thread_rank() < (block.size() - 32) ? thread_role = cuda::pipeline_role::consumer : cuda::pipeline_role::producer;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, SHMEM_CHUNK_SIZE> shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state);
    
    if (thread_role == cuda::pipeline_role::producer)
    {
        pipeline.prodcer_acquire();
        for (int i = block.thread_rank(); i < SHMEM_CHUNK_SIZE; i += block.size())
        {
            shmem[i] = src[i];
        }
        pipeline.producer_commit();
    }
}









int main()
{
    const unsigned int size = std::pow(2, 24);
    std::vector<float> host_src(size), host_dst(size);
    float* device_src, *device_dst;

    CUDA_CHECK(cudaMalloc(&device_src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_dst, size * sizeof(float)));
    std::fill(host_src.begin(), host_src.end(), 1);
    CUDA_CHECK(cudaMemcpy(device_src, host_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    const unsigned int num_threads = 512;
    const unsigned int num_blocks = 1;
    const unsigned int SHMEM_SIZE = 32768;

    dim3 grid(num_blocks);
    dim3 block(num_threads);
    KernelLogger logger("async");
    increment<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
    for (int i = 0; i < 10; i++)
    {
        logger.Start();
        increment<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
        logger.Stop();
    }



    cudaMemcpy(host_dst.data(), device_dst, size * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    for (unsigned int i = 0; i < size; i++)
    {
        if (host_dst[i] != host_src[i] + 1)
        {
            std::cerr << "Mismatch at index: " << i << " expected: " << host_src[i] + 1 << " got: " << host_dst[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }




}