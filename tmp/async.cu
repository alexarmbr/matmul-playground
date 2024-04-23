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

__device__ void produce(float* src,
     float* shmem,
     const unsigned int num_elements,
     const size_t shmem_offsets[2],
     barrier ready[2],
     barrier filled[2]
    )
{
    const unsigned int chunk_size = shmem_offsets[1];
    for (unsigned int i=0; i < (num_elements / chunk_size); i++)
    {
        ready[i%2].arrive_and_wait();
        float* shmem_chunk = shmem + shmem_offsets[i%2];
        for (unsigned int j = threadIdx.x; j < chunk_size; j+=32)
        {
            shmem_chunk[j] = src[i*chunk_size + j];
        }
        barrier::arrival_token token = filled[i%2].arrive();
    }

}

__device__ void consume(float* shmem,
     float* dst,
     unsigned int num_elements,
     size_t shmem_offsets[2],
     barrier ready[2],
     barrier filled[2]
    )
{
    const unsigned int chunk_size = shmem_offsets[1];
    barrier::arrival_token token = ready[0].arrive();
    token = ready[1].arrive();
    for (unsigned int i=0; i < (num_elements / chunk_size); i++)
    {
        filled[i%2].arrive_and_wait();
        float* shmem_chunk = shmem + shmem_offsets[i%2];
        for (unsigned int j = threadIdx.x - 32; j < chunk_size; j+=blockDim.x-32)
        {
            for (unsigned int k = 0; k < 10; k++)
            {
                shmem_chunk[j]+=1;
            }
            dst[i * chunk_size + j] = shmem_chunk[j];
        }
        barrier::arrival_token token = ready[i%2].arrive();
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

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier[4];
    if (block.thread_rank() < 4)
    {
        init(barrier + block.thread_rank(), block.size());
    }
    block.sync();

    if (block.thread_rank() < 32)
    {
        produce(src, shmem, num_elements, shmem_offset, barrier, barrier + 2);
    }
    else
    {
        consume(shmem, dst, num_elements, shmem_offset, barrier, barrier + 2);
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
    //     if (host_dst[i] != host_src[i] + 10)
    //     {
    //         std::cerr << "Mismatch at index: " << i << " expected: " << host_src[i] + 10 << " got: " << host_dst[i] << std::endl;
    //         exit(EXIT_FAILURE);
    //     }
    // }

    std::cout << "Avg Kernel Time: " << logger.getAvgTime() << " ms" << std::endl;
}