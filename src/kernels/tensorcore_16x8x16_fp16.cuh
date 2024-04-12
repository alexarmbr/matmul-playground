#pragma once
#include <cuda.h>

__global__ void tensorcore_16x8x16(half* A,
    half* B,
    half* C,
    half* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    unsigned int K)
{   
    // 1d block, 1 warp
    assert(blockDim.x == 32);
    assert(threadIdx.y == 0);
    assert(M == 16);
    assert(N == 8);
    assert(K == 16);

    __shared__ half A_shared[16][16];
    int row = threadIdx.x / K;
    int col = threadIdx.x % K;
    const int row_step = blockDim.x / K;
    while (row < M) {
        A_shared[row][col] = A[row * K + col];
        row += row_step;
    }

    uint32_t R[4];
    const int fragment_row = threadIdx.x % 16;
    const int fragment_col = threadIdx.x / 16;

    uint32_t *smem_ptr = reinterpret_cast<uint32_t*>(A_shared) + fragment_row * 8 + fragment_col * 4;
    asm(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(*smem_ptr)
    );
    
    const int thread_offset = threadIdx.x * 4;
    D[thread_offset] = R[0];
    D[thread_offset + 1] = R[1];
    D[thread_offset + 2] = R[2];
    D[thread_offset + 3] = R[3];

    // asm(
    //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    //     " {%0, %1}, " // two registers for D
    //     " {%2, %3, %4, %5}, " // four registers for A
    //     " {%6, %7}, " // two registers for B
    //     " {%8, %9} ;" // two registers for C
    //     : "=r"(D[??]), "=r"(D[??])
    //     : "r"(A[??]), "r"(A[??]), "r"(A[??]), "r"(A[??]),
    //       "r"(B[??]), "r"(B[??]),
    //       "r"(C[??]), "r"(C[??])
    // );
}

// how many registers for D, A, B, C
// for thread i, which values need to go to D, A, B, C
