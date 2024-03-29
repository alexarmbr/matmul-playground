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







// BM/BN/BK - dimension of blocktile per M/N/K (matrix dimensions). We must be able to fit 3
// tiles in shared memory: BMxBN for C, BMxBK for B, BKxBN for B, so shared memory capacity 
// per SM is the limiting factor here. 1 thread block handles each tile.
//
// WM/WN/WK - # of warp tiles per BM/BN/BK (block tile dimensions). 1 warp handles each tile
template<unsigned int BM,
unsigned int BN,
unsigned int BK,
unsigned int WM,
unsigned int WN,
unsigned int WK>
__global__ void fp32_Warptiling_Sgemm(float* A,
    float* B,
    float* C,
    float* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{
    // allocate shared memory using template params
    // calculate M,N indices for this thread block
    // write outer loop over k dimension, vectorized shmem loads
    // figure out thread/data mapping for shmem loading
    __shared__ float A_blocktile[BM * BK];
    __shared__ float B_blocktile[BK * BM];
    __shared__ float DC_blocktile[BM * BN];






}