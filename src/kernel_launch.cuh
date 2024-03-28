#pragma once
#include "util.cuh"
#include "kernels/fp16/tensorcore_naive.cuh"


void tensorcore_naive_launch(sgemm_params<half> device_sgemm_params)
{
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int M_TILES_PER_BLOCK = 2;
    const unsigned int N_TILES_PER_BLOCK = 4;
    const unsigned int TILE_DIM = 16;
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    // kernel setup and launch
    const unsigned int yBlocks = M / (M_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int xBlocks = N / (N_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int yThreadsPerBlock = M_TILES_PER_BLOCK;
    const unsigned int xThreadsPerBlock = WARP_SIZE * N_TILES_PER_BLOCK;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    tensorcore_naive_sgemm
    <M_TILES_PER_BLOCK, N_TILES_PER_BLOCK, TILE_DIM>
    <<<gridDim, blockDim>>>(
        device_sgemm_params.A,
        device_sgemm_params.B,
        device_sgemm_params.C,
        device_sgemm_params.D,
        device_sgemm_params.alpha,
        device_sgemm_params.beta,
        M,
        N,
        K
    );
    CUDA_CHECK(cudaPeekAtLastError());
}
