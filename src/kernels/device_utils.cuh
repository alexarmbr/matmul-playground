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



__device__ __forceinline__ void ldmatrix_m16n8(
    half* shmem,
    half (&reg)[4],
    const unsigned int shmem_stride_bytes
)
{
    uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    constexpr int frag_M_dim = 16;
    const unsigned int fragment_row = threadIdx.x % frag_M_dim;
    const unsigned int offset = fragment_row * shmem_stride_bytes;
    uint32_t* smem_ptr = reinterpret_cast<uint32_t*>(shmem) + offset;
    
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(reg_[0]), "=r"(reg_[1])
        : "r"(cvta_to_shared_u32(smem_ptr))
    );
}

__device__ __forceinline__ void ldmatrix_n8k8(
    half* shmem,
    half (&reg)[2],
    const unsigned int shmem_stride_bytes
)
{
    uint32_t &reg_ = reinterpret_cast<uint32_t&>(reg);
    constexpr int frag_K_dim = 8;
    const unsigned int fragment_row = threadIdx.x % frag_K_dim;
    const unsigned int offset = fragment_row * shmem_stride_bytes;
    uint32_t* smem_ptr = reinterpret_cast<uint32_t*>(shmem) + offset;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
        "{%0}, [%1];"
        : "=r"(reg_)
        : "r"(cvta_to_shared_u32(smem_ptr))
    );
}

__device__ __forceinline__ void mma_sync_m16n8k8(
    half (&D)[4],
    half (&A)[4],
    half (&B)[2],
    half (&C)[4]
)
{
    uint32_t (&D_)[2] = reinterpret_cast<uint32_t(&)[2]>(D);
    uint32_t (&A_)[2] = reinterpret_cast<uint32_t(&)[2]>(A);
    uint32_t (&C_)[2] = reinterpret_cast<uint32_t(&)[2]>(C);
    uint32_t &B_ = reinterpret_cast<uint32_t&>(B);

    asm volatile (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4}, "
        "{%5, %6};"
        : "=r"(D_[0]), "=r"(D_[1])
        : "r"(A_[0]), "r"(A_[1]),
          "r"(B_),
          "r"(C_[0]), "r"(C_[1])
    );
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
    const unsigned int M,
    const unsigned int N,
    const unsigned int K,
    bool accumulate_C
)
{
    constexpr unsigned int WARP_SIZE = 32;
    const unsigned int laneIdx = threadIdx.x % WARP_SIZE;

    // B is 8x8
    half B_register[2];
    ldmatrix_n8k8(B_shared, B_register, N / 2);

    // scale B by alpha
    half* B_register_half = reinterpret_cast<half*>(&B_register);
    B_register_half[0] *= alpha;
    B_register_half[1] *= alpha;

    // load A,C
    half A_register[4];
    half C_register[4];

    ldmatrix_m16n8(C_shared, C_register, K / 2);
    ldmatrix_m16n8(A_shared, A_register, K / 2);

    if (accumulate_C) {
        C_register[0] *= beta;
        C_register[1] *= beta;
        C_register[2] *= beta;
        C_register[3] *= beta;
    }
    

    // compute D
    half D_register[4];
    mma_sync_m16n8k8(D_register, A_register, B_register, C_register);
    
    // TODO rewrite ptr arithmetic for half precision
    
    // uint32_t* gmem_ptr_D = reinterpret_cast<uint32_t*>(D);
    // int fragment_row = laneIdx / 4;
    // const int fragment_col = laneIdx % 4;
    // gmem_ptr_D[fragment_row * 4 + fragment_col] = D_register[0];
    // int fragment_row_2 = fragment_row + 8;
    // gmem_ptr_D[fragment_row_2 * 4 + fragment_col] = D_register[1];
}

