#include "batched_gemm.cuh"

#include "cutlass/gemm/device/gemm_batched.h"

class CutlassBatchedGemmm : public BatchedGemm {
    using GemmOp = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor
        >;
    



    
private:
    GemmOp gemm_op;
}