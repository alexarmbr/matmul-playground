#include <assert.h>

#include "host_utils.cuh"
#include "kernel_launch.cuh"

#include <vector>

// int main(int argc, char **argv) {
//     bool check_on_cpu = false;

//     KernelLogger timer("Cublas");
//     std::vector<unsigned int> matrix_dims = {128, 512, 1024, 2048, 4096};
//     for (unsigned int D : matrix_dims)
//     {
//         auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(D, D, D);
//         cublas_fp16_launch(device_sgemm_params, timer);
//         if (check_on_cpu) {
//             sgemm_verify(device_sgemm_params, host_sgemm_params);
//         }
//     }

    
//     return 0;
//   }


  int main(int argc, char **argv) {
    bool check_on_cpu = true;

    KernelLogger timer("Tensorcore_2");
    const unsigned int M = 16;
    const unsigned int N = 16;
    const unsigned int K = 32;
    // for (unsigned int D : matrix_dims)
    // {
        auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(M, N, K);
        // CUDA_CHECK(cudaMemset(device_sgemm_params.D, 0, M * N * sizeof(half)));
        // memset(host_sgemm_params.D, 0, M * N * sizeof(half));
        // CUDA_CHECK(cudaMemset(device_sgemm_params.A, 0, M * K * sizeof(half)));
        // memset(host_sgemm_params.A, 0, M * K * sizeof(half));
        // CUDA_CHECK(cudaMemset(device_sgemm_params.B, 0, K * N * sizeof(half)));
        // memset(host_sgemm_params.B, 0, K * N * sizeof(half));
        device_sgemm_params.alpha = 1.0f;
        host_sgemm_params.alpha = 1.0f;
        device_sgemm_params.beta = 0.0f;
        host_sgemm_params.beta = 0.0f;

        tensorcore_2_launch(device_sgemm_params, timer);
        if (check_on_cpu) {
            sgemm_verify(device_sgemm_params, host_sgemm_params);
        }
    // }
    
    return 0;
  }