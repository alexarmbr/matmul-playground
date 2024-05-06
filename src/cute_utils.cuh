#pragma once
#include <cute/tensor.hpp>
using namespace cute;

constexpr unsigned int int_log2(unsigned int x)
{
    unsigned int result = 0;
    while (x >>= 1)
    {
        result++;
    }
    return result;
}

template<class Engine, class Layout>
__device__ void inspect_tensor(Tensor<Engine, Layout> T, const char *name = "")
{
    if (name != "")
    {
        printf("%s\n", name);
    }

    printf("threadIdx.x: %d, threadIdx.y: %d, blockIdx.x: %d, blockIdx.y: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    print(T.layout());
    printf("\n");
    for (int i = 0; i < size<0>(T); i++)
    {
      for (int j = 0; j < size<1>(T); j++)
      {
        printf("%0.2f ", (float) T(i, j));
      }
      printf("\n");
    }
}

template <int ROWS, int COLS, class TensorSrc, class TensorDst>
__device__ void tileMemcpyTranspose(TensorSrc src, TensorDst dst, const unsigned int src_stride_elements)
{
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(src.data()), make_shape(ROWS, COLS/8), make_stride(src_stride_elements / 8, 1));
    Tensor dst_float4 = make_tensor(reinterpret_cast<float4*>(dst.data().get()), make_shape(COLS/8, ROWS), LayoutLeft{});
    while (thread_idx < src_float4.size())
    {
      dst_float4(thread_idx) = src_float4(thread_idx);
      thread_idx += num_threads;
    }
}

template <int ROWS, int COLS, int SWIZZLE_BITS, class TensorSrc, class TensorDst>
__device__ void tileMemcpySwizzle(TensorSrc src, TensorDst dst, const unsigned int src_stride_elements)
{
    assert(COLS % 8 == 0);
    // // assert(SWIZZLE_BITS == (int) log2(COLS/8));
    // if (SWIZZLE_BITS != (int) log2(COLS/8))
    // {
    //   printf("COLS: %d, SWIZZLE_BITS: %d, log2(COLS/8): %d\n", COLS, SWIZZLE_BITS, (int) log2(COLS/8));
    //   assert(false);
    // }
    
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(src.data()), make_shape(ROWS, COLS/8), make_stride(src_stride_elements / 8, 1));
    auto swizzled_layout = composition(Swizzle<3,0,SWIZZLE_BITS>{}, make_layout(make_shape(ROWS, COLS/8), make_stride(COLS/8, 1)));
    Tensor dst_float4 = make_tensor(reinterpret_cast<float4*>(dst.data().get()), swizzled_layout);
    // if (thread_idx == 0 & blockIdx.x == 0 & blockIdx.y == 0 && SWIZZLE_BITS == 3)
    // {
    //   printf("src_float4 layout:\n");
    //   print_layout(src_float4.layout());
    //   printf("\n");
    //   printf("dst_float4 layout:\n");
    //   print_layout(dst_float4.layout());
    //   printf("\n");
    // }
    
    while (thread_idx < src_float4.size())
    {
      // auto src_index = crd2idx(thread_idx, src_float4.shape(), src_float4.stride());
      // auto dst_index = crd2idx(thread_idx, dst_float4.shape(), dst_float4.stride());
      dst_float4(thread_idx) = src_float4(thread_idx);
      thread_idx += num_threads;
    }
}


<template class Tensor>
__device__ __forceinline__ void ldmatrix_m16n8(
  Tensor T,
  half (&reg)[4],
)
{
  shmem_stride_bytes /= sizeof(uint32_t);
  uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
  constexpr int frag_M_dim = 16;
  const unsigned int fragment_row = threadIdx.x % frag_M_dim;
  const unsigned int offset = fragment_row * shmem_stride_bytes;
  uint32_t* smem_ptr = reinterpret_cast<uint32_t*>(shmem) + offset;
  
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(reg_[0]), "=r"(reg_[1])
      : "r"(cvta_to_shared_u32(smem_ptr))
  );
}

__device__ __forceinline__ void ldmatrix_n8k8(
  half* shmem,
  half (&reg)[2],
  unsigned int shmem_stride_bytes
)
{
  shmem_stride_bytes /= sizeof(uint32_t);
  uint32_t &reg_ = reinterpret_cast<uint32_t&>(reg);
  constexpr int frag_K_dim = 8;
  const unsigned int fragment_row = threadIdx.x % frag_K_dim;
  const unsigned int offset = fragment_row * shmem_stride_bytes;
  uint32_t* smem_ptr = reinterpret_cast<uint32_t*>(shmem) + offset;

  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
      "{%0}, [%1];"
      : "=r"(reg_)
      : "r"(cvta_to_shared_u32(smem_ptr))
  );
}