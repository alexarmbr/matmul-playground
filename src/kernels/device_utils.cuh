#pragma once
#include <cuda.h>



__device__ void tileMemcpySwizzle(
    half* src,
    half* dst,
    const unsigned int src_stride_bytes,
    const unsigned int dst_stride_bytes,
    const unsigned int TILE_ROWS
)
{
    // one row contains 64 halfs, 128 bytes, 32 words
    // so all values in each column fall on the same memory bank
    constexpr unsigned int TILE_COLS = 64;
    assert(src_stride_bytes % 128 == 0);
    assert(dst_stride_bytes % 128 == 0);

    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_float4 = src_stride_bytes / sizeof(float4);
    const unsigned int dst_stride_float4 = dst_stride_bytes / sizeof(float4);

    const unsigned int lane = threadIdx.x % 32;
    const unsigned int src_col = lane % 8;
    unsigned int src_row = lane / 8;

    unsigned int dst_row = (src_col & 1) | ((src_col >> 1) & 2);
    const unsigned int dst_col = ((src_col << 1) & 4) | (src_row ^ dst_row);

    while (src_row < TILE_ROWS)
    {
        dst_float4[dst_row * dst_stride_float4 + dst_col] = src_float4[src_row * src_stride_float4 + src_col];
        src_row += 4;
        dst_row += 4;
    }
}


// copy from a tile of shape (row, col) to a tile of shape (col, row)
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS>
__device__ void tileMemcpyTranspose(
    half* src,
    half* dst,
    const unsigned int src_stride_bytes,
    const unsigned int dst_stride_bytes
)
{
    // copy/transpose is performed in 16x16 tiles
    // reading 2 rows of 8 halfs each == 2x16 bytes is the smallest chunk
    // of global memory we can read without wasting any bandwidth
    // a 16x16 tile (128 bytes) can be written to a row of shmem with 0 bank conflicts
    static_assert(TILE_COLS % 16 == 0);
    static_assert(TILE_ROWS % 16 == 0);
    assert(src_stride_bytes % (TILE_COLS * sizeof(half)) == 0);
    assert(dst_stride_bytes % (TILE_ROWS * sizeof(half)) == 0);
    assert(blockDim.y == 1);

    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_float4 = src_stride_bytes / sizeof(float4);
    const unsigned int dst_stride_float4 = TILE_ROWS;
    constexpr unsigned int tile_cols_float4 = TILE_COLS / (sizeof(float4) / sizeof(half));
    constexpr unsigned int COL_STRIDE = 2;
    const unsigned int ROW_STRIDE = blockDim.x / COL_STRIDE;

    // adjacent threads go down rows, do two columns at a time
    unsigned int src_row = threadIdx.x % ROW_STRIDE;
    while (src_row < TILE_ROWS)
    {
        unsigned int src_col = threadIdx.x / ROW_STRIDE;
        while (src_col < tile_cols_float4)
        {
            dst_float4[src_col * dst_stride_float4 + src_row] = src_float4[src_row * src_stride_float4 + src_col];
            src_col += 2;
        }
        src_row += ROW_STRIDE;
    }
}

// copy from a tile of shape (row, col) to a tile of shape (col, row)
// template<unsigned int TILE_ROWS,
// unsigned int TILE_COLS,
// unsigned int MMA_TILE_ROWS,
// unsigned int MMA_TILE_COLS>
// __device__ void tileMemcpyTranspose2(
//     half* src,
//     half* dst,
//     const unsigned int src_stride_bytes,
//     const unsigned int dst_stride_bytes
// )
// {

//     constexpr unsigned int TILE_ROWS_ = TILE_ROWS;
//     constexpr unsigned int TILE_COLS_ = TILE_COLS;
//     constexpr unsigned int MMA_TILE_ROWS_ = MMA_TILE_ROWS;
//     constexpr unsigned int MMA_TILE_COLS_ = MMA_TILE_COLS;
    
//     constexpr unsigned int WARP_TILE_ROWS = 16;
//     constexpr unsigned int WARP_TILE_COLS_FLOAT4 = 2;
//     constexpr unsigned int MMA_TILE_COLS_FLOAT4 = MMA_TILE_COLS / 8;
//     constexpr unsigned int TILE_COLS_FLOAT4 = TILE_COLS / 8;
//     assert(blockDim.y == 1);
//     assert(MMA_TILE_COLS_FLOAT4 == 1);
//     assert(TILE_ROWS % WARP_TILE_ROWS == 0);

//     float4* src_float4 = reinterpret_cast<float4*>(src);
//     float4* dst_float4 = reinterpret_cast<float4*>(dst);
//     const unsigned int src_stride_float4 = src_stride_bytes / sizeof(float4);
//     const unsigned int dst_stride_float4 = dst_stride_bytes / sizeof(float4);
    
//     // each warp handles 2 (16x8) tiles per step
//     constexpr unsigned int num_warp_tiles_column = TILE_COLS_FLOAT4 / WARP_TILE_COLS_FLOAT4;
//     constexpr unsigned int num_warp_tiles_row = TILE_ROWS / WARP_TILE_ROWS;

//     const unsigned int warp_idx = threadIdx.x / 32;
//     const unsigned int warp_row_idx = warp_idx / num_warp_tiles_column;
//     const unsigned int warp_col_idx = warp_idx % num_warp_tiles_column;
    
//     // relative to block tile
//     const unsigned int warp_row = warp_row_idx * WARP_TILE_ROWS;
//     const unsigned int warp_col = warp_col_idx * WARP_TILE_COLS_FLOAT4;

//     const unsigned int thread_lane = threadIdx.x % 32;
//     const unsigned int thread_row = thread_lane % 16;
//     const unsigned int thread_col = thread_lane / 16;

//     // relative to warp tile
//     const unsigned int mma_tile_row = (thread_row / MMA_TILE_ROWS) * MMA_TILE_ROWS; 
//     const unsigned int mma_tile_col = thread_col;

//     // how many columns does the warp span
//     const unsigned int float4_columns_per_block = blockDim.x / 16;
//     const unsigned int rows_per_block = WARP_TILE_ROWS / ((float4_columns_per_block + TILE_COLS_FLOAT4 - 1) / TILE_COLS_FLOAT4);
//     const unsigned int dst_rows_per_block = rows_per_block / 16;
    
//     // how many times do we need to advance threads across columns/rows
//     unsigned int current_src_row = warp_row + thread_row;
//     unsigned int current_dst_row = warp_row_idx * (WARP_TILE_ROWS / MMA_TILE_ROWS) + (thread_row / MMA_TILE_ROWS);
//     const unsigned int dst_row_stride = TILE_COLS_FLOAT4 * 8;
//     const unsigned int dst_col_stride = 8;
//     while (current_src_row < TILE_ROWS)
//     {
//         unsigned int current_src_col = warp_col + thread_col;
//         unsigned int current_dst_col = current_src_col;
//         while (current_src_col < TILE_COLS_FLOAT4)
//         {
//             dst_float4[current_dst_row * dst_row_stride + current_dst_col * dst_col_stride] = \
//             src_float4[current_src_row * src_stride_float4 + current_src_col];
//             current_src_col += float4_columns_per_block;
//             current_dst_col += float4_columns_per_block;
//         }
//         current_src_row += rows_per_block;
//         current_dst_row += dst_rows_per_block;
//     }
// }



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
    unsigned int shmem_stride_bytes
)
{
    shmem_stride_bytes /= sizeof(uint32_t);
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
    unsigned int shmem_stride_bytes
)
{
    shmem_stride_bytes /= sizeof(uint32_t);
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

// the stmatrix ptx instruction works for sm_90 and above
// this is a workaround
__device__ __forceinline__ void stmatrix_m16n8(
    half* dst,
    half (&reg)[4],
    unsigned int dst_stride_bytes
)
{
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
    dst_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;
    
    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[0];
    fragment_row += 8;
    dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[1];
}

// the stmatrix ptx instruction works for sm_90 and above
// this is a workaround
__device__ __forceinline__ void ldmatrix_m16n8_gmem(
    half* src,
    half (&reg)[4],
    unsigned int src_stride_bytes
)
{
    const unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
    src_stride_bytes /= sizeof(uint32_t);
    unsigned int fragment_row = laneIdx / 4;
    const unsigned int fragment_col = laneIdx % 4;
    
    // 4 adjacent threads storing 4 bytes each == 16 byte transactions
    reg_[0] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
    fragment_row += 8;
    reg_[1] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
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

}

