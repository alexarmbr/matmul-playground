#include <assert.h>

#include "host_utils.cuh"
#include "kernel_launch.cuh"

#include <vector>

int main(int argc, char **argv) {
    bool check_on_cpu = false;

    KernelLogger timer("Cublas");
    std::vector<unsigned int> matrix_dims = {128, 512, 1024, 2048, 4096};
    for (unsigned int D : matrix_dims)
    {
        auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(D, D, D);
        cublas_fp16_launch(device_sgemm_params, timer);
        if (check_on_cpu) {
            sgemm_verify(device_sgemm_params, host_sgemm_params);
        }
    }

    
    return 0;
  }