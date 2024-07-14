#pragma once
#include "cuda_runtime.h"
#include <cuda_fp16.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <assert.h>

#include "structs_n_stuff.cuh"

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


// generate a random half precision float between LO and HI
inline half RAND_HALF(float LO = -1.0f, float HI = 1.0f)
{
    float r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    return (half) r;
}




std::pair<sgemm_params, sgemm_params> sgemm_setup(unsigned int M, unsigned int N, unsigned int K, float alpha = 0.7, float beta = 0.3)
{
    // setup
    half *A, *B, *C, *D;
    A = (half *)malloc(M * K * sizeof(half));
    B = (half *)malloc(K * N * sizeof(half));
    C = (half *)malloc(M * N * sizeof(half));
    D = (half *)malloc(M * N * sizeof(half));

    // allocate device matrices
    half *dev_A, *dev_B, *dev_C, *dev_D;
    CUDA_CHECK(cudaMalloc((void **)&dev_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void **)&dev_D, M * N * sizeof(half)));
    half LO = 0.0f;
    half HI = 1.0f;

    // fill host matrices with random elements
    srand(1234);
    for (int i = 0; i < M * N; i++) {
      C[i] = RAND_HALF();
      // C[i] = (half) i;
    }
    for (int i = 0; i < K * N; i++)
    {
      B[i] = RAND_HALF();
      // B[i] = (half) 1.0;
    }
    for (int i = 0; i < M * K; i++)
    {
      A[i] = RAND_HALF();
      // A[i] = (half)((0.1f * (float) i) / 256.0f);
    }
    
    // copy to device
    CUDA_CHECK(cudaMemcpy(dev_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_C, C, M * N * sizeof(half), cudaMemcpyHostToDevice));

    sgemm_params device_sgemm_params = {dev_A, dev_B, dev_C, dev_D, alpha, beta, M, N, K};
    sgemm_params host_sgemm_params = {A, B, C, D, alpha, beta, M, N, K};
    return std::make_pair(device_sgemm_params, host_sgemm_params);
}

void host_sgemm(sgemm_params params)
{
    half *A = params.A;
    half *B = params.B;
    half *C = params.C;
    half *D = params.D;
    half alpha = params.alpha;
    half beta = params.beta;
    unsigned int M = params.M;
    unsigned int N = params.N;
    unsigned int K = params.K;

    for (int m = 0; m < M; m++)
    {
    for (int n = 0; n < N; n++)
    {
        
        half acc = 0.0f;
        for (int k = 0; k < K; k++)
        {
        acc += (A[m * K + k] * B[k * N + n]);
        }
        D[m * N + n] = alpha * acc + (beta * C[m * N + n]);
    }
    }
}


bool elementwise_isclose(half* a, half* b, int size, float atol = 0.5)
{
    for (int i = 0; i < size; i++)
    {
        if (std::abs((float) a[i] - (float) b[i]) > atol)
        {
            std::cout << "Mismatch at index " << i << ": " << (float) a[i] << " != " << (float) b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void sgemm_verify(sgemm_params device_sgemm_params, sgemm_params host_sgemm_params)
{
    const unsigned int M = host_sgemm_params.M;
    const unsigned int N = host_sgemm_params.N;
    half *D = (half *)malloc(M * N * sizeof(half));
    CUDA_CHECK(cudaMemcpy(D, device_sgemm_params.D, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    host_sgemm(host_sgemm_params);
    assert(elementwise_isclose(D, host_sgemm_params.D, M * N));
}
