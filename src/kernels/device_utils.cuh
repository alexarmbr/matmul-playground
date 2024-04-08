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
    const unsigned int iterations = (TILE_COLS * TILE_ROWS) / blockDim.x;
    assert(iterations * blockDim.x == TILE_COLS * TILE_ROWS);
    assert(threadIdx.y == 0);

    const unsigned int row = threadIdx.x / TILE_COLS;
    const unsigned int col = threadIdx.x - (row * TILE_COLS);
    const unsigned int thread_step = blockDim.x / TILE_COLS;

    src += row * src_stride + col;
    dst += row * dst_stride + col;
    for (unsigned int step = 0; step < iterations; step++)
    {
        *dst = *src;
        src += (thread_step * src_stride);
        dst += (thread_step * dst_stride);
    }
}