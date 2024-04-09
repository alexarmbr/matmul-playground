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
    bool check_on_cpu = false;

    KernelLogger timer("Tensorcore_2");
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;
    
    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(M, N, K);
    tensorcore_1_launch(device_sgemm_params, timer);
    if (check_on_cpu) {
        sgemm_verify(device_sgemm_params, host_sgemm_params);
    }
    
    return 0;
  }