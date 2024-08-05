#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <thread>

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


struct sgemm_params
{
  half* A;
  half* B;
  half* C;
  half* D;
  float alpha;
  float beta;
  unsigned int M;
  unsigned int N;
  unsigned int K;
};


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
            // std::ofstream file;
            // file.open(loggerName + ".csv");
            // file << "info, gflops_per_sec\n";
            // for (auto log : logs)
            // {
            //       file << log.first << ", " << log.second << "\n";
            // }
            // file.close();
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
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

      double getAvgTime()
      {
            return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
      }
};

