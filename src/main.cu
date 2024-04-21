#include <assert.h>

#include "host_utils.cuh"
#include "kernel_launch.cuh"

#include <vector>

  int main(int argc, char **argv) {
    bool check_on_cpu = false;

    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <kernel_id> <num_iterations>" << std::endl;
        return 1;
    }

    const unsigned int kernel_id = std::stoi(argv[1]);
    std::string timer_name = "kernel_" + std::to_string(kernel_id);
    const unsigned int num_iterations = std::stoi(argv[2]);

    KernelLogger timer(timer_name);
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;
    
    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(M, N, K);
    // device_sgemm_params.beta = 1.0f;
    // host_sgemm_params.beta = 1.0f;
    switch (kernel_id) {
        case 1:
            tensorcore_1_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 2:
            tensorcore_2_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 3:
            tensorcore_3_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 4:
            tensorcore_4_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 5:
            // device_sgemm_params.alpha = 0.0f;
            // host_sgemm_params.alpha = 0.0f;
            tensorcore_5_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 6:
            // device_sgemm_params.alpha = 0.0f;
            // host_sgemm_params.alpha = 0.0f;
            tensorcore_6_launch(device_sgemm_params, timer, num_iterations);
            break;
        
        case 8:
            memcpy_launch(device_sgemm_params, timer, num_iterations);
        case 9:
            tensorcore_m16n8k8_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 10:
            // device_sgemm_params.alpha = 0.0f;
            // device_sgemm_params.beta = 1.0f;
            // host_sgemm_params.alpha = 0.0f;
            // host_sgemm_params.beta = 1.0f;
            tensorcore_tile_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 11:
            cublas_fp16_launch(device_sgemm_params, timer, num_iterations);
            break;
    }
    
    if (check_on_cpu) {
        sgemm_verify(device_sgemm_params, host_sgemm_params);
    }
    
    return 0;
  }