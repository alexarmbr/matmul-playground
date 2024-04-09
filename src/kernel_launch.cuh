#pragma once
#include "host_utils.cuh"
#include "kernels/tensorcore_1.cuh"
#include "kernels/tensorcore_2.cuh"
#include "kernels/fp32_blocktiling.cuh"

#include <cublas_v2.h>


#define NUM_RUNS 10

void tensorcore_1_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 2;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 2;
    constexpr unsigned int BM_dim = WM_dim * WARPS_PER_BLOCK_M;
    constexpr unsigned int BN_dim = WN_dim * WARPS_PER_BLOCK_N;
    constexpr unsigned int BK_dim = WK_dim * WARPS_PER_BLOCK_K; 
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    const unsigned int ThreadsM = 1;
    const unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);

    // warmup
    tensorcore_1
    <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    
    // for (int i = 0; i < NUM_RUNS; i++)
    // {
    //     timer.Start();
    //     tensorcore_1
    //     <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>
    //     <<<gridDim, blockDim>>>(
    //         device_sgemm_params.A,
    //         device_sgemm_params.B,
    //         device_sgemm_params.C,
    //         device_sgemm_params.D,
    //         device_sgemm_params.alpha,
    //         device_sgemm_params.beta,
    //         M,
    //         N,
    //         K
    //     );
    //     timer.Stop();
    // }
    // double gflops_per_sec = timer.logKernelStats(M, N, K);
    // std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    // CUDA_CHECK(cudaPeekAtLastError());
}


void tensorcore_2_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 2;
    constexpr unsigned int WARPS_PER_BLOCK_N = 2;
    constexpr unsigned int WARPS_PER_BLOCK_K = 2;
    constexpr unsigned int BM_dim = WM_dim * WARPS_PER_BLOCK_M;
    constexpr unsigned int BN_dim = WN_dim * WARPS_PER_BLOCK_N;
    constexpr unsigned int BK_dim = WK_dim * WARPS_PER_BLOCK_K; 
    const unsigned int M = device_sgemm_params.M;
    const unsigned int N = device_sgemm_params.N;
    const unsigned int K = device_sgemm_params.K;

    assert(M % BM_dim == 0);
    assert(N % BN_dim == 0);
    assert(K % BK_dim == 0);
    
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int BlocksM = M / BM_dim;
    const unsigned int BlocksN = N / BN_dim;
    const unsigned int ThreadsM = 1;
    const unsigned int ThreadsN = WARP_SIZE * WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);

    tensorcore_2
    <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // for (int i = 0; i < NUM_RUNS; i++)
    // {
    //     timer.Start();
    //     tensorcore_2
    //     <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim>
    //     <<<gridDim, blockDim>>>(
    //         device_sgemm_params.A,
    //         device_sgemm_params.B,
    //         device_sgemm_params.C,
    //         device_sgemm_params.D,
    //         device_sgemm_params.alpha,
    //         device_sgemm_params.beta,
    //         M,
    //         N,
    //         K
    //     );
    //     timer.Stop();
    // }
    // double gflops_per_sec = timer.logKernelStats(M, N, K);
    // std::cout << "Tiled TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    // CUDA_CHECK(cudaPeekAtLastError());

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

    // for (int i = 0; i < NUM_RUNS; i++)
    // {
    //     timer.Start();
    //     // warmup
    //     cublasStatus_t status = cublasHgemm(handle,
    //         CUBLAS_OP_N,
    //         CUBLAS_OP_N,
    //         M,
    //         N,
    //         K,
    //         &alpha,
    //         device_sgemm_params.A,
    //         K,
    //         device_sgemm_params.B,
    //         N,
    //         &beta,
    //         device_sgemm_params.C,
    //         N
    //     );
    //     timer.Stop();

    //     if (status != CUBLAS_STATUS_SUCCESS)
    //     {
    //         throw std::runtime_error("cuBLAS kernel failed");
    //     }
    // }
    // double gflops_per_sec = timer.logKernelStats(M, N, K);
    // std::cout << "cuBLAS: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
}

