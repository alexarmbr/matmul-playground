#include <assert.h>
#include "host_utils.cuh"

    // void kernel_1_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_2_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_3_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_4_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_5_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_6_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_7_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_8_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    // void kernel_9_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void cublas_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);

  int main(int argc, char **argv) {
    bool check_on_cpu = false;
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <kernel_id> <num_iterations>" << std::endl;
        return 1;
    }

    const unsigned int kernel_id = std::stoi(argv[1]);
    std::string timer_name = "kernel_" + std::to_string(kernel_id);
    const unsigned int num_iterations = std::stoi(argv[2]);
    assert(num_iterations > 0);

    KernelLogger timer(timer_name);
    // const unsigned int M = 256;
    // const unsigned int N = 256;
    // const unsigned int K = 256;
    const unsigned int M = 4096;
    const unsigned int N = 4096;
    const unsigned int K = 4096;
    
    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup(M, N, K);
    switch (kernel_id) {
        // case 1:
        //     kernel_1_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        // case 2:
        //     kernel_2_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        // case 3:
        //     kernel_3_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        // case 4:
        //     kernel_4_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        // case 5:
        //     kernel_5_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        case 6:
            kernel_6_launch(device_sgemm_params, timer, num_iterations);
            break;
        // case 7:
        //     kernel_7_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        case 8:
            kernel_8_launch(device_sgemm_params, timer, num_iterations);
            break;
        // case 9:
        //     kernel_9_launch(device_sgemm_params, timer, num_iterations);
        //     break;
        case 99:
            cublas_launch(device_sgemm_params, timer, num_iterations);
            break;

    }
    
    if (check_on_cpu) {
        sgemm_verify(device_sgemm_params, host_sgemm_params);
    }
    
    return 0;
  }