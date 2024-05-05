#pragma once
#include <cute/tensor.hpp>
using namespace cute;

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

template <int ROWS, int COLS, class TensorSrc, class TensorDst>
__device__ void tileMemcpySwizzle(TensorSrc src, TensorDst dst, const unsigned int src_stride_elements)
{
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(src.data()), make_shape(ROWS, COLS/8), make_stride(src_stride_elements / 8, 1));
    const int swizzle_bits = log2(COLS/8);
    auto swizzled_layout = composition(Swizzle<3,0,4>{}, src_float4.layout());
    Tensor dst_float4 = make_tensor(reinterpret_cast<float4*>(dst.data().get()), swizzled_layout);
    
    while (thread_idx < src_float4.size())
    {
      dst_float4(thread_idx) = src_float4(thread_idx);
      thread_idx += num_threads;
    }
}
