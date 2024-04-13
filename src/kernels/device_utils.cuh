#pragma once
#include <cuda.h>


// load TILE_ROWS * TILE_COLS from src into dst
// assumes 1d theadblock, i.e. threadIdx.y always equals 0
// iterations is the # of times we need to iterate, passed
// as a parameter so that each thread isnt computing the same
// value. It is ceil((TILE_ROWS * TILE_COLS) / blockDim.x)

// TODO there needs to be a dst_stride argument
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
typename T>
__device__ void tileMemcpy(
    T* src,
    T* dst,
    const unsigned int src_stride,
    const unsigned int dst_stride
)
{
    // assert(row_iterations * column_iterations * blockDim.x == TILE_ROWS * TILE_COLS);
    assert(threadIdx.y == 0);
    
    const unsigned int row_step = max(1, blockDim.x / TILE_COLS);
    const unsigned int col_step = blockDim.x;
    
    // const unsigned int column_iterations = min(1, TILE_COLS / col_step);
    // const unsigned int row_iterations = TILE_ROWS / row_step;

    const unsigned int thread_row = threadIdx.x / TILE_COLS;
    const unsigned int thread_col = threadIdx.x - (thread_row * TILE_COLS);
    
    for (unsigned int r = thread_row; r < TILE_ROWS; r+=row_step)
    {
        for (unsigned int c = thread_col; c < TILE_COLS; c+=col_step)
        {
            dst[r * dst_stride + c] =  src[r * src_stride + c];
        }
    }
    
}

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



// a warp does a single mma operation with A,B,C coming from shared memory
// D is written back to global memory
//
// D = alpha * A *       B + beta * C
// (16x8) =    (16x8) * (8x8) +     (16x8)
__device__ __forceinline__ void mma_m16n8k8(
    half* A_shared,
    half* B_shared,
    half* C_shared,
    half* D, // D is a pointer to global memory
    half alpha,
    half beta,
    const unsigned int A_shared_stride,
    const unsigned int B_shared_stride,
    const unsigned int C_shared_stride,
    const unsigned int D_stride,
    bool accumulate_C
)
{
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int laneIdx = threadIdx.x % WARP_SIZE;

    // D, A, C are 16x8
    constexpr int frag_M = 16;
    constexpr int fragment_stride_bytes = 4;

    // A/C are 16x8
    uint32_t A_register[2];
    uint32_t C_register[2];
    uint32_t* smem_ptr_A;
    uint32_t* smem_ptr_C;
    {
        const int fragment_row = laneIdx % frag_M;
        const int offset = fragment_row * fragment_stride_bytes;
        smem_ptr_A = reinterpret_cast<uint32_t*>(A_shared) + offset;
        smem_ptr_C = reinterpret_cast<uint32_t*>(C_shared) + offset;
    }
    
    // B is 8x8
    uint32_t B_register;
    uint32_t* smem_ptr_B;
    {
        const int fragment_row = laneIdx % frag_M;
        const int offset = fragment_row * fragment_stride_bytes;
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

    // scale B by alpha
    half* B_register_half = reinterpret_cast<half*>(&B_register);
    B_register_half[0] *= alpha;
    B_register_half[1] *= alpha;

    
    if (accumulate_C)
    {
        // load C
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
            "{%0, %1}, [%2];"
            : "=r"(C_register[0]), "=r"(C_register[1])
            : "r"(cvta_to_shared_u32(smem_ptr_C))
        );
        // scale C by beta
        half* C_register_half = reinterpret_cast<half*>(C_register);
        C_register_half[0] *= beta;
        C_register_half[1] *= beta;
        C_register_half[2] *= beta;
        C_register_half[3] *= beta;
    }
    else
    {
        half* C_register_half = reinterpret_cast<half*>(C_register);
        C_register_half[0] = 0;
        C_register_half[1] = 0;
        C_register_half[2] = 0;
        C_register_half[3] = 0;
    }

    // compute D
    uint32_t D_register[2];
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
    
    uint32_t* gmem_ptr_D = reinterpret_cast<uint32_t*>(D);
    int fragment_row = laneIdx / 4;
    const int fragment_col = laneIdx % 4;
    gmem_ptr_D[fragment_row * fragment_stride_bytes + fragment_col] = D_register[0];
    int fragment_row_2 = fragment_row + 8;
    gmem_ptr_D[fragment_row_2 * fragment_stride_bytes + fragment_col] = D_register[1];
}

