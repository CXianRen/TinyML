#include "MNorm.hpp"
#include "test_common.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_MNorm(){
    START_TEST();
    auto gamma = load_data<FP_T>(
      "build/test/temp/gamma.bin", 1 * 768);
    auto beta = load_data<FP_T>(
      "build/test/temp/beta.bin", 1 * 768);
    auto input = load_data<FP_T>(
      "build/test/temp/input.bin", 1 * 5 * 768);
    auto output = load_data<FP_T>(
      "build/test/temp/output.bin", 1 * 5 * 768);
    
    MLayerNorm<FP_T> norm(768, 1e-5f);
    norm.fill_gamma(gamma.data(), 1 * 768);
    norm.fill_beta(beta.data(), 1 * 768);

    Tensor<FP_T> input_tensor({1, 5, 768}, input);

    Tensor<FP_T> result_tensor = norm.forward(input_tensor);

    compare_data<FP_T>(result_tensor.data().get(), 
                        output.data(), 1 * 5 * 768, 1e-5f);

    PASSLOG();
}

int main(int argc, char** argv) {
    test_MNorm();
    return 0;
}