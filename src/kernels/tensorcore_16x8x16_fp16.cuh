#pragma once
#include <cuda.h>
#include "device_utils.cuh"

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
  }

__global__ void tensorcore_m16n8k8(half* A,
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
    assert(K == 8);

    __shared__ half A_shared[16 * 8];
    __shared__ half B_shared[8 * 8];
    __shared__ half C_shared[16 * 8];
    __shared__ half D_shared[16 * 8];

    // load A and B into shared memory
    tileMemcpy<16, 8, half>(A, A_shared, K, 8);
    tileMemcpy<8, 8, half>(B, B_shared, N, 8);
    tileMemcpy<16, 8, half>(C, C_shared, N, 8);

    
    // D, A, C are 16x8
    uint32_t D_register[2];
    uint32_t A_register[2];
    uint32_t C_register[2];
    uint32_t* smem_ptr_D;
    uint32_t* smem_ptr_A;
    uint32_t* smem_ptr_C;
    {
        const int fragment_row = threadIdx.x % 16;
        const int offset = fragment_row * 4;
        smem_ptr_A = reinterpret_cast<uint32_t*>(A_shared) + offset;
        smem_ptr_C = reinterpret_cast<uint32_t*>(C_shared) + offset;
        smem_ptr_D = reinterpret_cast<uint32_t*>(D_shared) + offset;
    }
    
    // B is 8x8
    uint32_t B_register;
    uint32_t* smem_ptr_B;
    {
        const int fragment_row = threadIdx.x % 8;
        const int offset = fragment_row * 4;
        smem_ptr_B = reinterpret_cast<uint32_t*>(B_shared) + offset;
    }

    // load A
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(A_register[0]), "=r"(A_register[1])
        : "r"(cvta_to_shared_u32(smem_ptr_A))
    );

    // load B
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
        "{%0}, [%1];"
        : "=r"(B_register)
        : "r"(cvta_to_shared_u32(smem_ptr_B))
    );

    // load C
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(C_register[0]), "=r"(C_register[1])
        : "r"(cvta_to_shared_u32(smem_ptr_C))
    );
    
    
    // const int thread_offset = threadIdx.x * 4;
    // half* A_register_ptr = reinterpret_cast<half*>(A_register);
    // half* B_register_ptr = reinterpret_cast<half*>(&B_register);
    // half* C_register_ptr = reinterpret_cast<half*>(C_register);

    // half a0 = A_register_ptr[0];
    // half a1 = A_register_ptr[1];
    // half a2 = A_register_ptr[2];
    // half a3 = A_register_ptr[3];
    
    // half b0 = B_register_ptr[0];
    // half b1 = B_register_ptr[1];
    
    // half c0 = C_register_ptr[0];
    // half c1 = C_register_ptr[1];
    // half c2 = C_register_ptr[2];
    // half c3 = C_register_ptr[3];
    // D[threadIdx.x] = a0;

    // compute D = 
    asm volatile (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        " {%0, %1}, " // two registers for D
        " {%2, %3}, " // two registers for A
        " {%4}, " // one registers for B
        " {%5, %6}; " // two registers for C
        : "=r"(D_register[0]), "=r"(D_register[1])
        : "r"(A_register[0]), "r"(A_register[1]),
          "r"(B_register),
          "r"(C_register[0]), "r"(C_register[1])
    );

    // store D
    uint32_t smem_ptr_D_ = cvta_to_shared_u32(smem_ptr_D);
    asm volatile (
        "stmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "[%0], {%1, %2};"
        : "=r"(smem_ptr_D_)
        : "r"(D_register[0]), "r"(D_register[1])
    );


    // half* D_register_ptr = reinterpret_cast<half*>(D_register);
    // half d0 = D_register_ptr[0];
    // half d1 = D_register_ptr[1];
    // D[threadIdx.x] = d0;
}

// how many registers for D, A, B, C
// for thread i, which values need to go to D, A, B, C
