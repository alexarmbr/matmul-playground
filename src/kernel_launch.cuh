#pragma once
#include "host_utils.cuh"
#include "kernels/tensorcore_1.cuh"
#include "kernels/tensorcore_2.cuh"
#include "kernels/fp32_blocktiling.cuh"

#include <cublas_v2.h>


#define NUM_RUNS 10

void tensorcore_1_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer)
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
    tensorcore_1
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
    
    for (int i = 0; i < NUM_RUNS; i++)
    {
        timer.Start();
        tensorcore_1
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
    double gflops_per_sec = timer.logKernelStats(M, N, K);
    std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    CUDA_CHECK(cudaPeekAtLastError());
}


void tensorcore_2_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer)
{

}








void cublas_fp16_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const half alpha = device_sgemm_params.alpha;
    const half beta = device_sgemm_params.beta;
    const int M = device_sgemm_params.M;
    const int N = device_sgemm_params.N;
    const int K = device_sgemm_params.K;

    // warmup
    cublasStatus_t status = cublasHgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        device_sgemm_params.A,
        K,
        device_sgemm_params.B,
        N,
        &beta,
        device_sgemm_params.C,
        N
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("cuBLAS kernel failed");
    }

    for (int i = 0; i < NUM_RUNS; i++)
    {
        timer.Start();
        // warmup
        cublasStatus_t status = cublasHgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M,
            N,
            K,
            &alpha,
            device_sgemm_params.A,
            K,
            device_sgemm_params.B,
            N,
            &beta,
            device_sgemm_params.C,
            N
        );
        timer.Stop();

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("cuBLAS kernel failed");
        }
    }
    double gflops_per_sec = timer.logKernelStats(M, N, K);
    std::cout << "cuBLAS: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
}

