class BatchedGemm {
public:
    virtual ~BatchedGemm() {}

    virtual void multiply(const float* A, const float* B, float* C, int numMatrices, int matrixSize) = 0;
};
