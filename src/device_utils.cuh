#pragma once
#include <cuda.h>
#include <assert.h>
#include <iostream>

/////////////////////////////////////////////////////////
// progressively more optimized versions of tileMemcpy //
/////////////////////////////////////////////////////////

// reasonable first implementation, coalesced gmem reads, bank conflict free writes
__device__ __forceinline__ void tileMemcpy(
    half* src,
    half* dst,
    const unsigned int src_stride,
    const unsigned int tile_rows,
    const unsigned int tile_cols
)
{
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;
    
    // # of threads is multiple of # of columns in the tile
    assert(num_threads % tile_cols == 0);
    
    // assign each thread a row/column in the tile, calculate the column step
    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;
    
    for (unsigned int r = thread_row; r < tile_rows; r+=row_step)
    {
        dst[r * tile_cols + thread_col] =  src[r * src_stride + thread_col];
    }
}

// same as above but with loop unrolled
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolled(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    // # of threads is multiple of # of columns in the tile
    static_assert(NUM_THREADS % TILE_COLS == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS;
    const unsigned int thread_col = thread_idx % TILE_COLS;
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        dst[thread_row * TILE_COLS + thread_col] =  src[thread_row * src_stride + thread_col];
        thread_row += ROW_STEP;
    }
    
}

// same as above but with vectorized reads/writes
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolledVectorized(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_vectorized = src_stride / 8;

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        dst_float4[thread_row * TILE_COLS_VECTORIZED + thread_col] =  src_float4[thread_row * src_stride_vectorized + thread_col];
        thread_row += ROW_STEP;
    }
    
}


// same as above, but writes are swizzled to avoid bank conflicts when shared memory is read later in the kernel
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS>
__device__ __forceinline__ void tileMemcpySwizzle(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    const unsigned int src_stride_vectorized = src_stride / 8;

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        dst_float4[dst_index] =  src_float4[src_index];
        thread_row += ROW_STEP;
    }
}




// same as above, performs only a load into registers, not the store into shared memory
// this function does not get stalled by memory latency
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpySwizzleLoad(
    half* src,
    float4 dst_reg[ELEMENTS_PER_THREAD],
    const unsigned int src_stride
)
{
    constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

    // reinterpret input/output as float4
    float4* src_float4 = reinterpret_cast<float4*>(src);
    const unsigned int src_stride_vectorized = src_stride / 8;

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // apply swizzle to the dst index
        const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        dst_reg[i] = src_float4[src_index];
        thread_row += ROW_STEP;
    }
}



template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpySwizzleStore(
    float4 src_reg[ELEMENTS_PER_THREAD],
    half* dst
)
{
    constexpr unsigned int SWIZZLE_MASK = 0b111 << SWIZZLE_BITS;

    // reinterpret input/output as float4
    float4* dst_float4 = reinterpret_cast<float4*>(dst);

    // # of threads is multiple of # of columns in the tile
    constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
    static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
    const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;
    
    // compile time check that we provided the right amount of registers for storage
    static_assert(ELEMENTS_PER_THREAD == NUM_ITERS);
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        // apply swizzle to the dst index
        unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
        dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);
        dst_float4[dst_index] = src_reg[i];
        thread_row += ROW_STEP;
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

__device__ __forceinline__ void ldmatrix_m16n16_gmem(
    half* src,
    half (&reg)[8],
    const unsigned int src_stride_bytes
)
{
    const unsigned int laneIdx = threadIdx.x % 32;
    float4 &reg_= reinterpret_cast<float4&>(reg);
    const unsigned int src_stride_float4 = src_stride_bytes / sizeof(float4);
    const float4* src_ptr = reinterpret_cast<float4*>(src);
    // const float4* thread_ptr = src_ptr + (thread_shmem_row_map[laneIdx] * src_stride_float4) + thread_shmem_col_map[laneIdx];
    const float4* thread_ptr = src_ptr + (laneIdx * src_stride_float4) + (laneIdx / 16);
    reg_ = *thread_ptr;
}


__device__ __forceinline__ void stmatrix_m16n16_gmem(
    half* dst,
    half (&reg)[8],
    const unsigned int dst_stride_bytes
)
{
    const unsigned int laneIdx = threadIdx.x % 32;
    float4 &reg_= reinterpret_cast<float4&>(reg);
    const unsigned int dst_stride_float4 = dst_stride_bytes / sizeof(float4);
    const float4* dst_ptr = reinterpret_cast<float4*>(dst);
    // const float4* thread_ptr = dst_ptr + (thread_shmem_row_map[laneIdx] * dst_stride_float4) + thread_shmem_col_map[laneIdx];
    const float4* thread_ptr = dst_ptr + (laneIdx * dst_stride_float4) + (laneIdx / 16);
    reg_ = *thread_ptr;
}



template <unsigned int BM_dim, unsigned int BK_dim>
__device__ void tileMemcpySwizzleUnrolled_A(half* src, half* dst, const unsigned int src_stride_half)
{
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr unsigned int BK_dim_float4 = BK_dim / 8;
    constexpr unsigned int TILE_SIZE = BM_dim * BK_dim_float4;
    const unsigned int src_stride_float4 = src_stride_half / 8;


    #pragma unroll 8
    while (thread_idx < TILE_SIZE)
    {
        const unsigned int thread_idx_y = thread_idx / BK_dim_float4;
        const unsigned int thread_idx_x = thread_idx % BK_dim_float4;
        const unsigned int src_ind = thread_idx_y * src_stride_float4 + thread_idx_x;
        unsigned int dst_ind = thread_idx_y * BK_dim_float4 + thread_idx_x;
        dst_ind = dst_ind ^ ((dst_ind & 0b10000) >> 4);
        dst_ind = dst_ind ^ ((dst_ind & 0b1100) >> 2);
        dst_float4[dst_ind] = src_float4[src_ind];
        thread_idx += num_threads;
    }
}

template <unsigned int BK_dim, unsigned int BN_dim>
__device__ void tileMemcpySwizzleUnrolled_B(half* src, half* dst, const unsigned int src_stride_half)
{
    float4* src_float4 = reinterpret_cast<float4*>(src);
    float4* dst_float4 = reinterpret_cast<float4*>(dst);
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    constexpr unsigned int BN_dim_float4 = BN_dim / 8;
    constexpr unsigned int TILE_SIZE = BK_dim * BN_dim_float4;
    const unsigned int src_stride_float4 = src_stride_half / 8;

    #pragma unroll 8
    while (thread_idx < TILE_SIZE)
    {
        const unsigned int thread_idx_y = thread_idx / BN_dim_float4;
        const unsigned int thread_idx_x = thread_idx % BN_dim_float4;
        const unsigned int src_ind = thread_idx_y * src_stride_float4 + thread_idx_x;
        unsigned int dst_ind = thread_idx_y * BN_dim_float4 + thread_idx_x;
        dst_ind = dst_ind ^ ((dst_ind & 0b1110000) >> 4);
        dst_float4[dst_ind] = src_float4[src_ind];
        thread_idx += num_threads;
    }
}


// useful functions
constexpr unsigned int int_log2(unsigned int x)
{
    unsigned int result = 0;
    while (x >>= 1)
    {
        result++;
    }
    return result;
}