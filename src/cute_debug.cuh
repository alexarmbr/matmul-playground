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

// template <int ROWS, int COLS>
// void tileMemcpyTranspose(half* src, half* dst)
// {
//     // assert(size<0>(src) == ROWS);
//     // assert(size<1>(src) == COLS);
//     // assert(size<1>(dst) == ROWS);
//     // assert(size<0>(dst) == COLS);
//     // assert(COLS % 4 == 0);
//     Tensor src_float4 = make_tensor(reinterpret_cast<float4*>(src.data()), make_shape(ROWS, COLS/4), LayoutRight{});
//     Tensor dst_float4 = make_tensor(reinterpret_cast<float4*>(dst.data()), make_shape(COLS/4, ROWS), LayoutRight{});
//     for (int i = 0; i < src_float4.size(); i++)
//     {
//         dst_float4(i) = src_float4(i);
//     }
// }
