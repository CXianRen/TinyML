#include "MEmbed.hpp"
#include "test_common.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_MEmbed(){
    START_TEST();
    auto weight = load_data<FP_T>("build/test/temp/weight.bin", 1000 * 768);
    auto input = load_data<int32_t>("build/test/temp/input.bin", 2 * 3);
    auto output = load_data<FP_T>("build/test/temp/output.bin", 2 * 3 * 768);
    MEmbed<FP_T> embed(1000, 768);
    embed.fill_weight(weight.data(), weight.size());

    Tensor<int32_t> input_tensor({2, 3}, input);

    Tensor<FP_T> result_tensor = embed.forward(input_tensor);

    compare_data<FP_T>(result_tensor.data().get(), 
                        output.data(), 2 * 3 * 768, 1e-5f);

    PASSLOG();
}

int main(int argc, char** argv) {
    test_MEmbed();
    return 0;
}