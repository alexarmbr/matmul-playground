#include <gtest/gtest.h>

#include "batched-gemm/helper.h"

// Test case 1
TEST(TestHelpers, TestCase1) {
    ASSERT_EQ(1, 1);

}

// Test case 2
TEST(TestHelpers, TestCase2) {
    ASSERT_EQ(2, 2);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
