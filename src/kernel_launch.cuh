#pragma once
#include "host_util.cuh"
#include "kernels/fp16/fp16_basic.cuh"


void tensorcore_naive_launch(sgemm_params<half> device_sgemm_params)
{
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BM = 2;
    const unsigned int BN = 4;
    const unsigned int TILE_DIM = 16;
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    // kernel setup and launch
    const unsigned int yBlocks = M / (BM * TILE_DIM);
    const unsigned int xBlocks = N / (BN * TILE_DIM);
    const unsigned int yThreadsPerBlock = BM;
    const unsigned int xThreadsPerBlock = WARP_SIZE * BN;
    static_assert((yThreadsPerBlock * xThreadsPerBlock / 32) == BM * BN, "# of warps in thread block must equal # of tiles in thread block");
    
    
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    tensorcore_naive_sgemm
    <BM, BN, TILE_DIM>
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
