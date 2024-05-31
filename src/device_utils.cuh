#pragma once
#include <cuda.h>
#include <assert.h>

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
    const unsigned int tile_cols_float4 = TILE_COLS / (sizeof(float4) / sizeof(half));

    // adjacent threads go down rows, do two columns at a time
    const unsigned int row_step = blockDim.x / tile_cols_float4;
    assert(blockDim.x % tile_cols_float4 == 0);
    unsigned int src_row = threadIdx.x / tile_cols_float4;
    const unsigned int src_col = threadIdx.x % tile_cols_float4;
    while (src_row < TILE_ROWS)
    {
        dst_float4[src_col * dst_stride_float4 + src_row] = src_float4[src_row * src_stride_float4 + src_col];
        src_row += row_step;
    }
}


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
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    
    const unsigned int row_step = max(1, num_threads / TILE_COLS);
    const unsigned int col_step = num_threads;
    
    // const unsigned int column_iterations = min(1, TILE_COLS / col_step);
    // const unsigned int row_iterations = TILE_ROWS / row_step;

    const unsigned int thread_row = thread_idx / TILE_COLS;
    const unsigned int thread_col = thread_idx - (thread_row * TILE_COLS);
    
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


__device__ __forceinline__ void ldmatrix_a(
    half* src,
    half (&reg)[4][8][4],
    const unsigned int smem_stride
)
{
    uint32_t (&reg_) [4][8][2] = reinterpret_cast<uint32_t(&)[4][8][2]>(reg);
    unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
    unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b111000000) >> 3);
    uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
    // when looking at this addr in debugger, it appears that it is just the number of bytes from the start of the shared memory

    constexpr int x_thread = 0;
    
    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][0][0]), "=r"(reg_[0][0][1]), "=r"(reg_[1][0][0]), "=r"(reg_[1][0][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][0][0]);
    // }
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][1][0]), "=r"(reg_[0][1][1]), "=r"(reg_[1][1][0]), "=r"(reg_[1][1][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][1][0]);
    // }

    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][2][0]), "=r"(reg_[0][2][1]), "=r"(reg_[1][2][0]), "=r"(reg_[1][2][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][2][0]);
    // }

    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][3][0]), "=r"(reg_[0][3][1]), "=r"(reg_[1][3][0]), "=r"(reg_[1][3][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][3][0]);
    // }

    src_addr ^= 0b1110000;

    // 4
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][4][0]), "=r"(reg_[0][4][1]), "=r"(reg_[1][4][0]), "=r"(reg_[1][4][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][4][0]);
    // }

    src_addr ^= 0b10000;

    // 5
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][5][0]), "=r"(reg_[0][5][1]), "=r"(reg_[1][5][0]), "=r"(reg_[1][5][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][5][0]);
    // }
    src_addr ^= 0b110000;
    
    // 6
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][6][0]), "=r"(reg_[0][6][1]), "=r"(reg_[1][6][0]), "=r"(reg_[1][6][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][6][0]);
    // }

    src_addr ^= 0b10000;

    // 7
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][7][0]), "=r"(reg_[0][7][1]), "=r"(reg_[1][7][0]), "=r"(reg_[1][7][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[0][7][0]);
    // }

    src_addr ^= 0b1000001110000;

    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][0][0]), "=r"(reg_[2][0][1]), "=r"(reg_[3][0][0]), "=r"(reg_[3][0][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][0][0]);
    // }
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][1][0]), "=r"(reg_[2][1][1]), "=r"(reg_[3][1][0]), "=r"(reg_[3][1][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][1][0]);
    // }

    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][2][0]), "=r"(reg_[2][2][1]), "=r"(reg_[3][2][0]), "=r"(reg_[3][2][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][2][0]);
    // }

    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][3][0]), "=r"(reg_[2][3][1]), "=r"(reg_[3][3][0]), "=r"(reg_[3][3][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][3][0]);
    // }

    src_addr ^= 0b1110000;

    // 4
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][4][0]), "=r"(reg_[2][4][1]), "=r"(reg_[3][4][0]), "=r"(reg_[3][4][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][4][0]);
    // }

    src_addr ^= 0b10000;

    // 5
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][5][0]), "=r"(reg_[2][5][1]), "=r"(reg_[3][5][0]), "=r"(reg_[3][5][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][5][0]);
    // }
    src_addr ^= 0b110000;
    
    // 6
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][6][0]), "=r"(reg_[2][6][1]), "=r"(reg_[3][6][0]), "=r"(reg_[3][6][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][6][0]);
    // }

    src_addr ^= 0b10000;

    // 7
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[2][7][0]), "=r"(reg_[2][7][1]), "=r"(reg_[3][7][0]), "=r"(reg_[3][7][1])
        : "r"(src_addr)
    );
    // if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
    //     printf("src_addr: %u\n", src_addr);
    //     printf("%f\n", (float) reg[2][7][0]);
    // }

}



__device__ __forceinline__ void ldmatrix_b(
    half* src,
    half (&reg)[8][8][2],
    const unsigned int smem_stride,
    half alpha

)
{
    uint32_t (&reg_) [8][8] = reinterpret_cast<uint32_t(&)[8][8]>(reg);
    unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
    unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b1111000000) >> 4);
    uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
    // when looking at this addr in debugger, it appears that it is just the number of bytes from the start of the shared memory

    constexpr int x_thread = 32;
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][0]), "=r"(reg_[1][0]), "=r"(reg_[2][0]), "=r"(reg_[3][0])
        : "r"(src_addr)
    );

    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][0][0]);
    }
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][1]), "=r"(reg_[1][1]), "=r"(reg_[2][1]), "=r"(reg_[3][1])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][1][0]);
    }

    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][2]), "=r"(reg_[1][2]), "=r"(reg_[2][2]), "=r"(reg_[3][2])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][2][0]);
    }

    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][3]), "=r"(reg_[1][3]), "=r"(reg_[2][3]), "=r"(reg_[3][3])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][3][0]);
    }

    src_addr ^= 0b1110000;

    // 4
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][4]), "=r"(reg_[1][4]), "=r"(reg_[2][4]), "=r"(reg_[3][4])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][4][0]);
    }

    src_addr ^= 0b10000;

    // 5
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][5]), "=r"(reg_[1][5]), "=r"(reg_[2][5]), "=r"(reg_[3][5])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][5][0]);
    }
    src_addr ^= 0b110000;
    
    // 6
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][6]), "=r"(reg_[1][6]), "=r"(reg_[2][6]), "=r"(reg_[3][6])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][6][0]);
    }

    src_addr ^= 0b10000;

    // 7
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[0][7]), "=r"(reg_[1][7]), "=r"(reg_[2][7]), "=r"(reg_[3][7])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[0][7][0]);
    }

    src_addr ^= 0b10000001110000;

    // 0
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][0]), "=r"(reg_[5][0]), "=r"(reg_[6][0]), "=r"(reg_[7][0])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][0][0]);
    }
    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][1]), "=r"(reg_[5][1]), "=r"(reg_[6][1]), "=r"(reg_[7][1])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][1][0]);
    }

    src_addr ^= 0b110000;

    // 2
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][2]), "=r"(reg_[5][2]), "=r"(reg_[6][2]), "=r"(reg_[7][2])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][2][0]);
    }

    src_addr ^= 0b10000;

    // 3
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][3]), "=r"(reg_[5][3]), "=r"(reg_[6][3]), "=r"(reg_[7][3])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][3][0]);
    }

    src_addr ^= 0b1110000;

    // 4
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][4]), "=r"(reg_[5][4]), "=r"(reg_[6][4]), "=r"(reg_[7][4])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][4][0]);
    }

    src_addr ^= 0b10000;

    // 5
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][5]), "=r"(reg_[5][5]), "=r"(reg_[6][5]), "=r"(reg_[7][5])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][5][0]);
    }
    src_addr ^= 0b110000;
    
    // 6
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][6]), "=r"(reg_[5][6]), "=r"(reg_[6][6]), "=r"(reg_[7][6])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][6][0]);
    }

    src_addr ^= 0b10000;

    // 7
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(reg_[4][7]), "=r"(reg_[5][7]), "=r"(reg_[6][7]), "=r"(reg_[7][7])
        : "r"(src_addr)
    );
    if (blockIdx.x == 0 && threadIdx.x == x_thread && threadIdx.y == 0) {
        printf("src_addr: %u\n", src_addr);
        printf("%f\n", (float) reg[4][7][0]);
    }

    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        #pragma unroll
        for (int n = 0; n < 8; n++)
        {
            reg[k][n][0] *= alpha;
            reg[k][n][1] *= alpha;
        }
    }

}