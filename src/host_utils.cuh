#pragma once
#include "cuda_runtime.h"
#include <cuda_fp16.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

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


// struct with generic type
template <typename T>
struct sgemm_params
{
  T* A;
  T* B;
  T* C;
  T* D;
  float alpha;
  float beta;
  unsigned int M;
  unsigned int N;
  unsigned int K;
};

template <typename T>
std::pair<sgemm_params<T>, sgemm_params<T>> sgemm_setup(unsigned int M, unsigned int N, unsigned int K, float alpha = 0.7, float beta = 0.3)
{
    // setup
    T *A, *B, *C, *D;
    A = (T *)malloc(M * K * sizeof(T));
    B = (T *)malloc(K * N * sizeof(T));
    C = (T *)malloc(M * N * sizeof(T));
    D = (T *)malloc(M * N * sizeof(T));

    // allocate device matrices
    T *dev_A, *dev_B, *dev_C, *dev_D;
    CUDA_CHECK(cudaMalloc((void **)&dev_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_C, M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_D, M * N * sizeof(T)));
    half LO = 0.0f;
    half HI = 1.0f;

    // fill host matrices with random elements
    srand(1234);
    for (int i = 0; i < M * N; i++) {
      C[i] = RAND_HALF();
      // C[i] = 0.0f;
    }
    for (int i = 0; i < K * N; i++)
    {
      B[i] = RAND_HALF();
      // B[i] = 1.0f;
    }
    for (int i = 0; i < M * K; i++)
    {
      // A[i] = RAND_HALF();
      A[i] = (half) i;
    }
    
    // copy to device
    CUDA_CHECK(cudaMemcpy(dev_A, A, M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B, K * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_C, C, M * N * sizeof(T), cudaMemcpyHostToDevice));

    sgemm_params<T> device_sgemm_params = {dev_A, dev_B, dev_C, dev_D, alpha, beta, M, N, K};
    sgemm_params<T> host_sgemm_params = {A, B, C, D, alpha, beta, M, N, K};
    return std::make_pair(device_sgemm_params, host_sgemm_params);
}


// template <>
void host_sgemm(sgemm_params<half> params)
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

// void host_sgemm(sgemm_params<float> params)
// {
//     float *A = params.A;
//     float *B = params.B;
//     float *C = params.C;
//     float *D = params.D;
//     float alpha = params.alpha;
//     float beta = params.beta;
//     unsigned int M = params.M;
//     unsigned int N = params.N;
//     unsigned int K = params.K;

//     for (int m = 0; m < M; m++)
//     {
//     for (int n = 0; n < N; n++)
//     {
        
//         float acc = 0.0f;
//         for (int k = 0; k < K; k++)
//         {
//         acc += (float) (A[m * K + k] * B[k * N + n]);
//         }
//         D[m * N + n] = alpha * acc + beta * C[m * N + n];
//     }
//     }
// }

template <typename T>
bool elementwise_isclose(T* a, T* b, int size, float atol = 1e-1)
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

template <typename T>
void sgemm_verify(sgemm_params<T> device_sgemm_params, sgemm_params<T> host_sgemm_params)
{
    const unsigned int M = host_sgemm_params.M;
    const unsigned int N = host_sgemm_params.N;
    T *D = (T *)malloc(M * N * sizeof(T));
    CUDA_CHECK(cudaMemcpy(D, device_sgemm_params.D, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    host_sgemm(host_sgemm_params);
    assert(elementwise_isclose(D, host_sgemm_params.D, M * N));
}

struct KernelLogger
{
      cudaEvent_t start;
      cudaEvent_t stop;
      std::vector<float> times;
      std::string loggerName;
      std::vector<std::pair<std::string, double>> logs;

      KernelLogger(std::string loggerName) : loggerName(loggerName)
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }

      ~KernelLogger()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // logs to csv file
            std::ofstream file;
            file.open(loggerName + ".csv");
            file << "info, gflops_per_sec\n";
            for (auto log : logs)
            {
                  file << log.first << ", " << log.second << "\n";
            }
            file.close();
      }

      void Start()
      {
            cudaEventRecord(start, 0);
      }

      void Stop()
      {
            cudaEventRecord(stop, 0);
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            times.push_back(elapsed);
      }
      double logKernelStats(const unsigned int M, const unsigned int N, const unsigned int K)
      {
        double avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double total_flops = 2.0 * M * N * K;
        double gflops_per_sec = (total_flops) / (avg_time_ms * 1.0e6);
        times.clear();
        std::string info = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        logs.push_back(std::make_pair(info, gflops_per_sec));
        return gflops_per_sec;
      }
};