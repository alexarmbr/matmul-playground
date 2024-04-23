#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <iostream>
#include <array>
#include <vector>
#include "structs_n_stuff.cuh"

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;

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
            
            for (int k = 0; k < 1000; k++)
            {
                shmem[j] += 1;
            }    
        }
        // __syncthreads();

        for (unsigned int j = threadIdx.x; j < SHMEM_SIZE; j+=blockDim.x)
        {
            dst[i+j] = shmem[j];
        }
    }
}



template <unsigned int SHMEM_SIZE_BYTES>
__global__ void increment_async(float* src, float* dst, unsigned int num_elements)
{
    extern __shared__ float shmem[];
    constexpr unsigned int SHMEM_SIZE = SHMEM_SIZE_BYTES / sizeof(float);
    constexpr unsigned int SHMEM_CHUNK_SIZE = SHMEM_SIZE / 2;
    size_t shmem_offset[2] = {0, SHMEM_CHUNK_SIZE};
    
    // fetch first chunk
    // for (unsigned int j = threadIdx.x; j < SHMEM_CHUNK_SIZE; j+=blockDim.x)
    // {
    //     shmem[j] = src[j];
    // }
    
    
    const unsigned int iterations = num_elements / SHMEM_CHUNK_SIZE;
    for (unsigned int chunk_i = 0; chunk_i <= iterations; chunk_i++)
    {
        // fetch i'th chunk
        if (chunk_i != iterations)
        {
            float* shmem_chunk = &shmem[shmem_offset[chunk_i % 2]];
            for (unsigned int j = threadIdx.x; j < SHMEM_CHUNK_SIZE; j+=blockDim.x)
            {
                shmem_chunk[j] = src[chunk_i * SHMEM_CHUNK_SIZE + j];
            }
        }
        
        // process i-1th chunk
        if (chunk_i != 0)
        {
            float* shmem_chunk = &shmem[shmem_offset[(chunk_i-1) % 2]];
            for (unsigned int j = threadIdx.x; j < SHMEM_CHUNK_SIZE; j+=blockDim.x)
            {
                for (unsigned int k = 0; k < 1000; k++)
                {
                    shmem_chunk[j]+=1;
                }
                dst[(chunk_i-1) * SHMEM_CHUNK_SIZE + j] = shmem_chunk[j];
            }
        }
    }




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
    
    
    increment<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
    
    for (int i = 0; i < 10; i++)
    {
        logger.Start();
        increment<SHMEM_SIZE><<<grid, block, SHMEM_SIZE>>>(device_src, device_dst, size);
        logger.Stop();
    }

    cudaMemcpy(host_dst.data(), device_dst, size * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    // for (unsigned int i = 0; i < size; i++)
    // {
    //     if (host_dst[i] != host_src[i] + 10)
    //     {
    //         std::cerr << "Mismatch at index: " << i << " expected: " << host_src[i] + 10 << " got: " << host_dst[i] << std::endl;
    //         exit(EXIT_FAILURE);
    //     }
    // }

    std::cout << "Avg Kernel Time: " << logger.getAvgTime() << " ms" << std::endl;
}