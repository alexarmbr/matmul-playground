#include <gtest/gtest.h>

#include "batched-gemm/helper.h"

// Test case 1
TEST(TestHelpers, TestAllocateMatrix) {

    const int M = 1024;
    const int N = 2048;
    const int batch = 128;

    float* A_host = new float[M * N * batch];
    float* A_device;
    CUDA_CHECK(AllocateMatrix(&A_device, batch, M, N));
    CUDA_CHECK(cudaMemcpy(A_host, A_device, sizeof(float) * M * N * batch, cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                // AllocateMatrix should be filling matrix with entry with the value
                // of the flattened index
                ASSERT_FLOAT_EQ(A_host[b * M * N + i * N + j], b * M * N + i * N + j);
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
