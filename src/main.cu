#include <assert.h>
#include "host_utils.cuh"

    void kernel_1_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_2_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_3_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_4_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_5_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void kernel_6_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);
    void cublas_launch(sgemm_params device_sgemm_params, KernelLogger& timer, const unsigned int num_runs);

  int main(int argc, char **argv) {
    bool check_on_cpu = false;
    
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <kernel_id> <num_iterations> <M> <N> <K>" << std::endl;
        return 1;
    }

    const unsigned int kernel_id = std::stoi(argv[1]);
    std::string timer_name = "kernel_" + std::to_string(kernel_id);
    const unsigned int num_iterations = std::stoi(argv[2]);
    const unsigned int M = std::stoi(argv[3]);
    const unsigned int N = std::stoi(argv[4]);
    const unsigned int K = std::stoi(argv[5]);
    assert(num_iterations > 0);

    KernelLogger timer(timer_name);

    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup(M, N, K);
    switch (kernel_id) {
        case 1:
            kernel_1_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 2:
            kernel_2_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 3:
            kernel_3_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 4:
            kernel_4_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 5:
            kernel_5_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 6:
            kernel_6_launch(device_sgemm_params, timer, num_iterations);
            break;
        case 99:
            cublas_launch(device_sgemm_params, timer, num_iterations);
            break;
    }
    
    if (check_on_cpu) {
        sgemm_verify(device_sgemm_params, host_sgemm_params);
    }
    
    return 0;
  }