#pragma once
#include "host_utils.cuh"
#include "kernels/fp16/fp16_basic.cuh"
#include "kernels/fp32/fp32_blocktiling.cuh"

#include <cublas_v2.h>
// #include "cutlass/gemm/device/gemm.h"


#define NUM_RUNS 10

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

    // warmup
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
    
    GpuTimer timer;
    for (int i = 0; i < NUM_RUNS; i++)
    {
        timer.Start();
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
        timer.Stop();
    }
    double time_ms = timer.getAvgTime();
    double gflops_per_sec = (2.0 * M * N * K) / (time_ms * 1.0e6);
    std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;


    CUDA_CHECK(cudaPeekAtLastError());
}

void fp32_blocktiling_launch(sgemm_params<float> device_sgemm_params)
{
    constexpr unsigned int BM = 32;
    constexpr unsigned int BN = 32;
    constexpr unsigned int BK = 64;
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    const unsigned int yBlocks = M / BM;
    const unsigned int xBlocks = N / BN;
    const unsigned int yThreadsPerBlock = 1;
    const unsigned int xThreadsPerBlock = BM * BN;

    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    fp32_Blocktiled_Sgemm
    <BM, BN, BK>
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



void cublas_fp32_launch(sgemm_params<float> device_sgemm_params)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, device_sgemm_params.M, device_sgemm_params.N, device_sgemm_params.K, &device_sgemm_params.alpha, device_sgemm_params.A, device_sgemm_params.M, device_sgemm_params.B, device_sgemm_params.K, &device_sgemm_params.beta, device_sgemm_params.C, device_sgemm_params.M);
    CUDA_CHECK(cudaPeekAtLastError());
    cublasDestroy(handle);
}

// void cutlass_fp32_launch(sgemm_params<float> device_sgemm_params)
// {
//     using Sgemm = cutlass::gemm::device::Gemm<
//     float,
//     cutlass::layout::RowMajor,
//     float,
//     cutlass::layout::RowMajor,
//     float,
//     cutlass::layout::RowMajor,
//     float>;

//     Sgemm sgemm_op;
//     Sgemm::Arguments args(
//         {device_sgemm_params.M, device_sgemm_params.N, device_sgemm_params.K},
//         {device_sgemm_params.A, device_sgemm_params.K},
//         {device_sgemm_params.B, device_sgemm_params.N},
//         {device_sgemm_params.C, device_sgemm_params.N},
//         {device_sgemm_params.D, device_sgemm_params.N},
//         {device_sgemm_params.alpha, device_sgemm_params.beta}
//     );
//     cutlass::Status status = sgemm_op(args);

//     if (status != cutlass::Status::kSuccess)
//     {
//         throw std::runtime_error("Cutlass kernel failed");
//     }
// }
