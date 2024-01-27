#include <iostream>

#include "cublasLt.h"

#include "batched-gemm/helper.h"
#include "batched-gemm/cuBlasLt_helper.h"


/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
void LtSgemm(
    int m,
    int n,
    int k,
    const float *A,
    const float *B,
    float *C,
    int num_trials) {
cublasLtMatmulDesc_t operationDesc = NULL;
cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
cublasLtMatmulPreference_t preference = NULL;

int returnedResults                             = 0;
cublasLtMatmulHeuristicResult_t heuristicResult = {};

cublasOperation_t transa = CUBLAS_OP_N;
cublasOperation_t transb = CUBLAS_OP_N;

cublasLtHandle_t ltHandle;
checkCublasStatus(cublasLtCreate(&ltHandle));
size_t workspaceSize = 4 * 1024 * 1024; // 32MiB
void* workspace;
checkCudaStatus(cudaMalloc((void **)&workspace, workspaceSize));
float alpha = 1.0f, beta = 0.0f;
int lda = m;
int ldb = k;
int ldc = m;
// std::cout << "m: " << m << std::endl;
// std::cout << "n: " << n << std::endl;
// std::cout << "k: " << k << std::endl;
// std::cout << "lda: " << lda << std::endl;
// std::cout << "ldb: " << ldb << std::endl;
// std::cout << "ldc: " << ldc << std::endl;

// create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
// set the transforms for A and B
checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

// create matrix descriptors, we are good with the details here so no need to set any extra attributes
checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, lda));
checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb));
checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

// create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
// will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
// directly come from cudaMalloc)
checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

// we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
// is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

if (returnedResults == 0) {
checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
}

for (int i = 0; i < num_trials; i++)
{
GpuTimer timer;
timer.start();
checkCublasStatus(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            A,
                            Adesc,
                            B,
                            Bdesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            0));
timer.stop();
std::cout << "elapsed ms: " << timer.elapsed_millis() << std::endl;
}
// descriptors are no longer needed as all GPU work was already enqueued
if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}




int main(int argc, char* argv[]) {
    // accept M, N, K, batch
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
    std::fill(C_host, C_host + M * N * batch, 0);

    float* A_device;
    float* B_device;
    float* C_device;
    CUDA_CHECK(AllocateMatrix(&A_device, batch, M, K));
    CUDA_CHECK(AllocateMatrix(&B_device, batch, K, N));
    CUDA_CHECK(AllocateMatrix(&C_device, batch, M, N));

    LtSgemm(M, N, K, A_device, B_device, C_device, 1);

    // CUDA_CHECK(cudaMemcpy(A_host, A_device, sizeof(float) * M * K * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(B_host, B_device, sizeof(float) * K * N * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(C_host, C_device, sizeof(float) * M * N * batch, cudaMemcpyDeviceToHost));

    // // TODO make sure these are correct for column major layout
    //     for (int i = 0; i < M; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             for (int k = 0; k < K; k++)
    //             {
    //                 // column major
    //                 C_host_ref[j * M + i] += A_host[k * M + i] * B_host[j * K + k];
    //             }
    //         }
    //     }

    // const float tolerance = 0.0001;
    //     for (int i = 0; i < M; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             if (std::abs(C_host_ref[j * M + i] - C_host[j * M + i]) > tolerance)
    //             {
    //                 std::cout << "Error at (" << i << ", " << j << ")" << std::endl;
    //                 std::cout << "Expected: " << C_host_ref[j * M + i] << std::endl;
    //                 std::cout << "Actual: " << C_host[j * M + i] << std::endl;
    //                 return 1;
    //             }
    //         }
    //     }
    




    // CUDA_CHECK(cudaMemcpy(A_host, A_device, sizeof(float) * M * K * batch, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(B_host, B_device, sizeof(float) * K * N * batch, cudaMemcpyDeviceToHost));




    return 0;
}
