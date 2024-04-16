#pragma once
#include "host_utils.cuh"
#include "kernels/tensorcore_1.cuh"
#include "kernels/tensorcore_2.cuh"
#include "kernels/tensorcore_3.cuh"
#include "kernels/tensorcore_4.cuh"
#include "kernels/memcpy.cuh"
#include "kernels/tensorcore_16x8x16_fp16.cuh"
#include "kernels/tensorcore_64x64x64.cuh"

#include <cublas_v2.h>

void tensorcore_1_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 2;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 1;
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
    
    if (num_runs != 0)
    {
        for (int i = 0; i < num_runs; i++)
        {
            timer.Start();
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
            timer.Stop();
        }
        double gflops_per_sec = timer.logKernelStats(M, N, K);
        std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
        CUDA_CHECK(cudaPeekAtLastError());
    }
}


void tensorcore_2_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 4;
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

    if (num_runs != 0)
    {
        for (int i = 0; i < num_runs; i++)
        {
            timer.Start();
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
            timer.Stop();
        }
        double gflops_per_sec = timer.logKernelStats(M, N, K);
        std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
        CUDA_CHECK(cudaPeekAtLastError());
    }

}


void memcpy_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 16;
    constexpr unsigned int WK_dim = 16;
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 4;
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

    memcpy
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

    if (num_runs != 0)
    {
        for (int i = 0; i < num_runs; i++)
        {
            timer.Start();
            memcpy
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
            timer.Stop();
        }
        double gflops_per_sec = timer.logKernelStats(M, N, K);
        std::cout << "Naive TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
        CUDA_CHECK(cudaPeekAtLastError());
    }

}




void tensorcore_3_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    
    constexpr unsigned int MMA_M_dim = 16;
    constexpr unsigned int MMA_N_dim = 16;
    constexpr unsigned int MMA_K_dim = 16;

    constexpr unsigned int MMA_TILES_PER_WARP_TILE_M=4;
    constexpr unsigned int MMA_TILES_PER_WARP_TILE_N=2;
    constexpr unsigned int MMA_TILES_PER_WARP_TILE_K=1;

    constexpr unsigned int WM_dim = MMA_M_dim * MMA_TILES_PER_WARP_TILE_M;
    constexpr unsigned int WN_dim = MMA_N_dim * MMA_TILES_PER_WARP_TILE_N;
    constexpr unsigned int WK_dim = MMA_K_dim * MMA_TILES_PER_WARP_TILE_K;

    constexpr unsigned int WARP_TILES_PER_BLOCK_M = 2;
    constexpr unsigned int WARP_TILES_PER_BLOCK_N = 2;
    constexpr unsigned int WARP_TILES_PER_BLOCK_K = 2; // is this needed?

    constexpr unsigned int BM_dim = WM_dim * WARP_TILES_PER_BLOCK_M;
    constexpr unsigned int BN_dim = WN_dim * WARP_TILES_PER_BLOCK_N;
    constexpr unsigned int BK_dim = WK_dim * WARP_TILES_PER_BLOCK_K;

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
    const unsigned int ThreadsN = WARP_SIZE * WARP_TILES_PER_BLOCK_M * WARP_TILES_PER_BLOCK_N;

    dim3 gridDim(BlocksN, BlocksM);
    dim3 blockDim(ThreadsN, ThreadsM);

    tensorcore_3
    <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, MMA_M_dim, MMA_N_dim, MMA_K_dim>
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

    if (num_runs != 0)
    {
        for (int i = 0; i < num_runs; i++)
        {
            timer.Start();
            tensorcore_3
            <BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, MMA_M_dim, MMA_N_dim, MMA_K_dim>
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
        std::cout << "Warp Tiled TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
        CUDA_CHECK(cudaPeekAtLastError());
    }

}


void tensorcore_m16n8k8_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    assert(device_sgemm_params.M == 16);
    assert(device_sgemm_params.N == 8);
    assert(device_sgemm_params.K == 8);
    
    dim3 gridDim(1);
    dim3 blockDim(32, 1);
    
    tensorcore_m16n8k8
    <<<gridDim, blockDim>>>(
        device_sgemm_params.A,
        device_sgemm_params.B,
        device_sgemm_params.C,
        device_sgemm_params.D,
        device_sgemm_params.alpha,
        device_sgemm_params.beta,
        device_sgemm_params.M,
        device_sgemm_params.N,
        device_sgemm_params.K
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

}

void tensorcore_m64n64k64_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    assert(device_sgemm_params.M == 64);
    assert(device_sgemm_params.N == 64);
    assert(device_sgemm_params.K == 64);
    
    dim3 gridDim(1);
    dim3 blockDim(32, 1);
    
    tensorcore_m64n64k64
    <<<gridDim, blockDim>>>(
        device_sgemm_params.A,
        device_sgemm_params.B,
        device_sgemm_params.C,
        device_sgemm_params.D,
        device_sgemm_params.alpha,
        device_sgemm_params.beta,
        device_sgemm_params.M,
        device_sgemm_params.N,
        device_sgemm_params.K
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}







void tensorcore_4_launch(sgemm_params<half> device_sgemm_params, KernelLogger& timer, const unsigned int num_runs = 10)
{
    constexpr unsigned int WM_dim = 16;
    constexpr unsigned int WN_dim = 8;
    constexpr unsigned int WK_dim = 8;
    
    constexpr unsigned int WARPS_PER_BLOCK_M = 4;
    constexpr unsigned int WARPS_PER_BLOCK_N = 4;
    constexpr unsigned int WARPS_PER_BLOCK_K = 4;
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

    tensorcore_4
    <BM_dim, BN_dim, BK_dim>
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

    if (num_runs != 0)
    {
        for (int i = 0; i < num_runs; i++)
        {
            timer.Start();
            tensorcore_4
            <BM_dim, BN_dim, BK_dim>
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
        std::cout << "mma TensorCore: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
        CUDA_CHECK(cudaPeekAtLastError());
    }
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

