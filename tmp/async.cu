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
    for (unsigned int i = 0; i < num_elements; i += SHMEM_SIZE)
    {
        for (unsigned int j = threadIdx.x; j < SHMEM_SIZE; j+=blockDim.x)
        {
            shmem[j] = src[i+j];
            
            for (int k = 0; k < 10; k++)
            {
                shmem[j] += 1;
            }    
        }
        __syncthreads();

        for (unsigned int j = threadIdx.x; j < SHMEM_SIZE; j+=blockDim.x)
        {
            dst[i+j] = shmem[j];
        }
    }
}

__device__ void produce(float* src, float* dst, unsigned int num_elements)
{
    if (threadIdx.x >= 32)
        return;

    for (unsigned int i = threadIdx.x; i < num_elements; i += 32)
    {
        dst[i] = src[i];
    }
}

__device__ void consume(float* src, float* dst, unsigned int num_elements)
{
    if (threadIdx.x < 32)
        return;

    for (unsigned int i = threadIdx.x - 32; i < num_elements; i++)
    {
        for (unsigned int k = 0; k < 10; k++)
        {
            src[i] += 1;
        }
        dst[i] = src[i];
    }
}

template <unsigned int SHMEM_SIZE_BYTES>
__global__ void increment_async(float* src, float* dst, unsigned int num_elements)
{
    extern __shared__ float shmem[];
    constexpr uint8_t PIPELINE_NUM_STAGES = 2;
    constexpr unsigned int SHMEM_SIZE = SHMEM_SIZE_BYTES / sizeof(float);
    constexpr unsigned int SHMEM_CHUNK_SIZE = SHMEM_SIZE / PIPELINE_NUM_STAGES;
    size_t shmem_offset[2] = {0, SHMEM_CHUNK_SIZE};

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    
    // threads in the first warp are producers, other threads are consumers
    cuda::pipeline_role thread_role =
     block.thread_rank() < 32 ? thread_role = cuda::pipeline_role::producer : cuda::pipeline_role::consumer;

    extern __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_NUM_STAGES> shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state);
    
    // if (thread_role == cuda::pipeline_role::producer)
    // {
        pipeline.producer_acquire();
        produce(src, shmem, SHMEM_CHUNK_SIZE);
        pipeline.producer_commit();
    // }
    // else
    // {
        pipeline.consumer_wait();
        consume(shmem, dst, SHMEM_CHUNK_SIZE);
        pipeline.consumer_release();
    // }


    // for (int chunk_i = 1; chunk_i < (num_elements / SHMEM_CHUNK_SIZE); chunk_i++)
    // {
    //     float* producer_chunk = &shmem[shmem_offset[chunk_i % 2]];
    //     float* consumer_chunk = &shmem[shmem_offset[(chunk_i - 1) % 2]];
    //     if (thread_role == cuda::pipeline_role::consumer)
    //     {
    //         if (threadIdx.x == 32)
    //         {
    //             printf("CONSUMER BEGIN\n");
    //         }
    //         pipeline.consumer_wait();
    //         consume(consumer_chunk, dst + (SHMEM_CHUNK_SIZE * (chunk_i-1)), SHMEM_CHUNK_SIZE);
    //         pipeline.consumer_release();
    //         if (threadIdx.x == 32)
    //         {
    //             printf("CONSUMER END\n");
    //         }
    //     }
    //     else // producer
    //     {
    //         if (threadIdx.x == 0)
    //         {
    //             printf("PRODUCER BEGIN\n");
    //         }
    //         pipeline.producer_acquire();
    //         produce(src + (SHMEM_CHUNK_SIZE * chunk_i), producer_chunk, SHMEM_CHUNK_SIZE);
    //         pipeline.producer_commit();
    //         if (threadIdx.x == 0)
    //         {
    //             printf("PRODUCER END\n");
    //         }
    //     }
    // }




}









int main()
{
    const unsigned int size = std::pow(2, 24);
    std::vector<float> host_src(size), host_dst(size);
    float* device_src, *device_dst;

    CUDA_CHECK(cudaMalloc(&device_src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_dst, size * sizeof(float)));
    for (int i = 0; i < size; i++)
    {
        host_src[i] = i;
    }

    CUDA_CHECK(cudaMemcpy(device_src, host_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    const unsigned int num_threads = 512;
    const unsigned int num_blocks = 1;
    const unsigned int SHMEM_SIZE = 32768;

    dim3 grid(num_blocks);
    dim3 block(num_threads);
    KernelLogger logger("async");
    increment_async<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
    // for (int i = 0; i < 10; i++)
    // {
    //     logger.Start();
    //     increment<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
    //     logger.Stop();
    // }

    cudaMemcpy(host_dst.data(), device_dst, size * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    // for (unsigned int i = 0; i < size; i++)
    // {
    //     if (host_dst[i] != host_src[i] + 1)
    //     {
    //         std::cerr << "Mismatch at index: " << i << " expected: " << host_src[i] + 1 << " got: " << host_dst[i] << std::endl;
    //         exit(EXIT_FAILURE);
    //     }
    // }

    // std::cout << "Avg Kernel Time: " << logger.getAvgTime() << " ms" << std::endl;
}