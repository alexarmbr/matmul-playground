#include <assert.h>

#include "host_util.cuh"
#include "kernel_launch.cuh"

#define M 256
#define N 128
#define K 64

int main(int argc, char **argv) {
    bool check_on_cpu = true;

    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>(M, N, K);

    tensorcore_naive_launch(device_sgemm_params);

    if (check_on_cpu) {
        sgemm_verify(device_sgemm_params, host_sgemm_params);
    }
    
    return 0;
  }