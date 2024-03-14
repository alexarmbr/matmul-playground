#include <iostream>
#include <bitset>
#include <batched-gemm/helper.h>
// this is the number of threads per block in the xy dimension
// the total number of threads per block is D * D
// and also the size of each side of the shared memory tile
// the total sizeof the shared memory tile is D * D * sizeof(float)
#define D 16

// for N=K=M=4096
//   my best kernel so far: ~310 ms
//   cublas: ~30 ms


void floatToBinary(float f)
{
    union { float f; uint32_t i; } u;
    u.f = f;
    std::string str;

    for (int i = 0; i < 32; i++)
    {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(str.rbegin(), str.rend());
    std::cout << temp << std::endl;
}


__global__ void matmul_kernel_naive(int M, int N, int K, float* A, float* B, float* C) {
    const unsigned int A_row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int B_col = blockDim.x * blockIdx.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
    {
        acc += A[A_row * K + k] * B[k * N + B_col];
    }
    C[A_row * N + B_col] = acc;
}


__global__ void matmul_kernel_shmem(int M, int N, int K, float* A, float* B, float* C) {
    __shared__ float tile_a [D*D];
    __shared__ float tile_b [D*D];

    const unsigned int A_row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int B_col = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int num_tiles = (K + D - 1) / D;
    const unsigned int tile_idx_flat = threadIdx.y * D + threadIdx.x;
    float local_sum = 0.0f;

    for (int k = 0; k < num_tiles; k++)
    {
        // load tiles of a and b and sync
        const unsigned int A_col = k * D + threadIdx.x;
        const unsigned int B_row = k * D + threadIdx.y;
        tile_a[tile_idx_flat] = A[A_row * K + A_col];
        tile_b[tile_idx_flat] = B[B_row * N + B_col];
        __syncthreads();

        // perform matmul from tile_a and tile_b into tile_c and sync
        for (int tile_k = 0; tile_k < D; tile_k++)
        {
            local_sum += tile_a[threadIdx.y * D + tile_k] * tile_b[tile_k * D + threadIdx.x];
        }
        __syncthreads();
    }
    C[A_row * N + B_col] = local_sum;

}

void matmul(int M, int N, int K, float* A, float* B, float* C, int num_trials) {

    // compute number of blocks to launch given M, N, K
    const int M_tiles = (M + D - 1) / D;
    const int N_tiles = (N + D - 1) / D;
    dim3 grid_dim(N_tiles, M_tiles, 1);
    dim3 block_dim(D, D, 1);

    for (int i = 0; i < num_trials; i++)
    {
        GpuTimer timer;
        timer.start();
        matmul_kernel_naive<<<grid_dim, block_dim>>>(M, N, K, A, B, C);
        CUDA_CHECK(cudaPeekAtLastError());
        timer.stop();
        std::cout << "elapsed ms: " << timer.elapsed_millis() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int batch = 1;
    if (argc == 5) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        batch = atoi(argv[4]);
    }

    float* A_host = new float[M * K * batch];
    float* B_host = new float[K * N * batch];
    float* C_host = new float[M * N * batch];
    float* C_host_ref = new float[M * N * batch];
    std::fill(C_host_ref, C_host_ref + M * N * batch, 0);

    float* A_device;
    float* B_device;
    float* C_device;
    CUDA_CHECK(AllocateMatrix(&A_device, batch, M, K));
    CUDA_CHECK(AllocateMatrix(&B_device, batch, K, N));
    CUDA_CHECK(AllocateMatrix(&C_device, batch, M, N));

    matmul(M, N, K, A_device, B_device, C_device, 10);

    // CUDA_CHECK(cudaMemcpy(A_host, A_device, sizeof(float) * M * K * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(B_host, B_device, sizeof(float) * K * N * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(C_host, C_device, sizeof(float) * M * N * batch, cudaMemcpyDeviceToHost));

    //     for (int i = 0; i < M; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             float acc = 0.0f;
    //             for (int k = 0; k < K; k++)
    //             {
    //                 acc += A_host[i * K + k] * B_host[k * N + j];
    //             }
    //             C_host_ref[i * N + j] =  acc;
    //         }
    //     }

    // const float tolerance = 0.01f;
    //     std::cout << "checking results at tolerance of " << tolerance << std::endl;
    //     for (int i = 0; i < M; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             if (std::abs(C_host_ref[j * M + i] - C_host[j * M + i]) > tolerance)
    //             {
    //                 std::cout << "Error at (" << i << ", " << j << ")" << std::endl;
    //                 std::cout << "Expected: " << C_host_ref[j * M + i] << std::endl;
    //                 std::cout << "Actual: " << C_host[j * M + i] << std::endl;
    //                 std::cout << "Difference: " << C_host_ref[j * M + i] - C_host[j * M + i] << std::endl;
    //                 // std::cout << "binary representation of expected: ";
    //                 // floatToBinary(C_host_ref[j * M + i]);
    //                 // std::cout << std::endl;
    //                 // std::cout << "binary representation of actual: ";
    //                 // floatToBinary(C_host[j * M + i]);
    //                 std::cout << std::endl;
    //                 return 1;
    //             }
    //         }
    //     }
    // CUDA_CHECK(cudaMemcpy(A_host, A_device, sizeof(float) * M * K * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(B_host, B_device, sizeof(float) * K * N * batch, cudaMemcpyDeviceToHost));
    return 0;
}
