#include "kernels/fp16/tensorcore_naive.cuh"
#include "util.cuh"

// struct with generic type
template <typename T>
struct sgemm_params
{
  T* A;
  T* B;
  T* C;
  T* D;
  float alpha;
  float beta;
};

template <typename T>
std::pair<sgemm_params<T>, sgemm_params<T>> sgemm_setup()
{
    // setup
    T *A, *B, *C, *D;
    A = (T *)malloc(M * K * sizeof(T));
    B = (T *)malloc(K * N * sizeof(T));
    C = (T *)malloc(M * N * sizeof(T));
    D = (T *)malloc(M * N * sizeof(T));
    float alpha = 0.3;
    float beta = 0.7;

    // allocate device matrices
    T *dev_A, *dev_B, *dev_C, *dev_D;
    CUDA_CHECK(cudaMalloc((void **)&dev_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_C, M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)&dev_D, M * N * sizeof(T)));

    // fill host matrices with random elements
    srand(1234);
    for (int i = 0; i < M * N; i++) {
      C[i] = (T)(rand() % 10);
    }
    for (int i = 0; i < K * N; i++)
    {
      B[i] = (T)(rand() % 10);
    }
    for (int i = 0; i < M * K; i++)
    {
      A[i] = (T)(rand() % 10);
    }
    
    // copy to device
    CUDA_CHECK(cudaMemcpy(dev_A, A, M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B, K * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_C, C, M * N * sizeof(T), cudaMemcpyHostToDevice));

    sgemm_params<T> device_sgemm_params = {dev_A, dev_B, dev_C, dev_D, alpha, beta};
    sgemm_params<T> host_sgemm_params = {A, B, C, D, alpha, beta};
    return std::make_pair(device_sgemm_params, host_sgemm_params);
}


// template <>
void host_sgemm(sgemm_params<half> params)
{
    half *A = params.A;
    half *B = params.B;
    half *C = params.C;
    half *D = params.D;
    float alpha = params.alpha;
    float beta = params.beta;
    for (int m = 0; m < M; m++)
    {
    for (int n = 0; n < N; n++)
    {
        
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
        {
        acc += (float) (A[m * K + k] * B[k * N + n]);
        }
        D[m * N + n] = alpha * acc + (float) ((half) beta * C[m * N + n]);
    }
    }
}

void host_sgemm(sgemm_params<float> params)
{
    throw std::runtime_error("Not implemented");
}

template <typename T>
bool elementwise_isclose(T* a, T* b, int size, float atol = 1e-5)
{
    for (int i = 0; i < size; i++)
    {
        if (std::abs((float) a[i] - (float) b[i]) > atol)
        {
            return false;
        }
    }
    return true;
}



int main(int argc, char **argv) {
    bool check_on_cpu = true;

    auto [device_sgemm_params, host_sgemm_params] = sgemm_setup<half>();

    // kernel setup and launch
    const unsigned int yBlocks = M / (M_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int xBlocks = N / (N_TILES_PER_BLOCK * TILE_DIM);
    const unsigned int yThreadsPerBlock = M_TILES_PER_BLOCK;
    const unsigned int xThreadsPerBlock = WARP_SIZE * N_TILES_PER_BLOCK;
    dim3 gridDim(xBlocks, yBlocks);
    dim3 blockDim(xThreadsPerBlock, yThreadsPerBlock);
    tensorcore_naive_sgemm<<<gridDim, blockDim>>>(
        device_sgemm_params.A,
        device_sgemm_params.B,
        device_sgemm_params.C,
        device_sgemm_params.D,
        device_sgemm_params.alpha,
        device_sgemm_params.beta
    );
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    // copy result back to host
    half *D = (half *)malloc(M * N * sizeof(half));
    CUDA_CHECK(cudaMemcpy(D, device_sgemm_params.D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    if (check_on_cpu) {
        host_sgemm(host_sgemm_params);
        elementwise_isclose(D, host_sgemm_params.D, M * N);
    }
    
        return 0;
  }