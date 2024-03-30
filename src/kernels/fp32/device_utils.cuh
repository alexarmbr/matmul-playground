#pragma once
#include <cuda.h>


// load TILE_ROWS * TILE_COLS from A_gmem into A_shared
// assumes 1d theadblock, i.e. threadIdx.y always equals 0
// iterations is the # of times we need to iterate, passed
// as a parameter so that each thread isnt computing the same
// value. It is ceil((TILE_ROWS * TILE_COLS) / blockDim.x)
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS>
__device__ void loadFromGmem(
    float* A_gmem,
    float* A_shared,
    const unsigned int A_stride
)
{
    // make sure blockDim.x divides TILE_COLS * TILE_ROWS
    // TODO move this calculation and assertion to host code
    const unsigned int iterations = (TILE_COLS * TILE_ROWS) / blockDim.x;
    assert(iterations * blockDim.x == TILE_COLS * TILE_ROWS);
    assert(threadIdx.y == 0);

    const unsigned int row = threadIdx.x / TILE_COLS;
    const unsigned int col = threadIdx.x - (row * TILE_COLS);
    const unsigned int thread_step = blockDim.x / TILE_COLS;

    A_gmem += row * A_stride + col;
    A_shared += row * TILE_COLS + col;
    for (unsigned int step = 0; step < iterations; step++)
    {
        *A_shared = *A_gmem;
        A_gmem += (thread_step * A_stride);
        A_shared += (thread_step * TILE_COLS);
    }
}