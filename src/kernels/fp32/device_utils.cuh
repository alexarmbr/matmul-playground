#pragma once
#include <cuda.h>


// load TILE_ROWS * TILE_COLS from M_gmem into M_shared
// assumes 1d theadblock, i.e. threadIdx.y always equals 0
// iterations is the # of times we need to iterate, passed
// as a parameter so that each thread isnt computing the same
// value. It is ceil((TILE_ROWS * TILE_COLS) / blockDim.x)
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS>
__device__ loadFromGmem(
    float* M_gmem,
    float* M_shared,
    const unsigned int M_stride,
)
{
    // blockDim.x divides (TILE_ROWS * TILE_COLS)
    const unsigned int iterations = (TILE_COLS * TILE_ROWS) / blockDim.x;
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    const unsigned int gmem_row = i / TILE_COLS;
    const unsigned int gmem_col = i - (i_row * TILE_COLS);

    const unsigned int shmem_row = threadIdx.x / TILE_COLS;
    const unsigned int shmem_col = threadIdx.x - (shmem_row * TILE_COLS);
    
    M_gmem += gmem_row * M_stride + gmem_col;
    M_shared += shmem_row * TILE_COLS + shmem_col;
    for (unsigned int step = 0; step < iterations; step++)
    {
        *M_shared = *M_gmem
        M_gmem += (gmem_row * M_stride);
        M_shared += (shmem_row * TILE_COLS);
    }
}