#pragma once
#include <cuda.h>


// load TILE_ROWS * TILE_COLS from src into dst
// assumes 1d theadblock, i.e. threadIdx.y always equals 0
// iterations is the # of times we need to iterate, passed
// as a parameter so that each thread isnt computing the same
// value. It is ceil((TILE_ROWS * TILE_COLS) / blockDim.x)

// TODO there needs to be a dst_stride argument
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
typename T>
__device__ void tileMemcpy(
    T* src,
    T* dst,
    const unsigned int src_stride,
    const unsigned int dst_stride
)
{
    // TODO work on this
    assert(threadIdx.y == 0);
    for (unsigned int row = 0; row < TILE_ROWS; row++)
    {
        for (unsigned int col = threadIdx.x; col < TILE_COLS; col += blockDim.x)
        {
            T* thread_src = src + row * src_stride + col;
            T* thread_dst = dst + row * dst_stride + col;
            *thread_dst = *thread_src;
        }
    }
}