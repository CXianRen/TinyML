#include "MActivation.hpp"
#include "test_common.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_MGELUActivation(){
    START_TEST();
    auto input = load_data<FP_T>(
      "build/test/temp/input.bin", 1 * 5 * 768);
    auto output = load_data<FP_T>(
      "build/test/temp/output.bin", 1 * 5 * 768);
    MGELUActivation<FP_T> at;
    
    Tensor<FP_T> input_tensor({1, 5, 768}, input);
    Tensor<FP_T> result_tensor = at.forward(input_tensor);

    compare_data<FP_T>(result_tensor.data().get(), 
                        output.data(), 1 * 5 * 768, 1e-5f);

    PASSLOG();
}

int main(int argc, char** argv) {
    test_MGELUActivation();
    return 0;
}