#include "MLinear.hpp"
#include "MTB.hpp"
#include "test_common.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_MLinear(){
    START_TEST();
    {
        MLinear<FP_T> linear(768, 768, false);
        auto input = mtb::random<FP_T>({2,5,768});
        auto output = linear.forward(input);
        assert(output.shape() == std::vector<int>({2, 5, 768}));
    }
    {
        MLinear<FP_T> linear(768, 768, true);
        auto input = mtb::random<FP_T>({2,5,768});
        auto output = linear.forward(input);
        assert(output.shape() == std::vector<int>({2, 5, 768}));
    }
    {
        auto weight = load_data<FP_T>("build/test/temp/weight.bin", 768*768);
        auto bias = load_data<FP_T>("build/test/temp/bias.bin", 768);
        auto input = load_data<FP_T>("build/test/temp/input.bin", 2*5*768);
        auto output = load_data<FP_T>("build/test/temp/output.bin", 2*5*768);

        MLinear<FP_T> linear(768, 768, true);
        //
        linear.fill_weight(weight.data(), weight.size());
        linear.fill_bias(bias.data(), bias.size());

        auto input_t = Tensor<FP_T>({2, 5, 768}, input);
        auto output_t = linear.forward(input_t);
        compare_data(output_t.data().get(), 
                    output.data(), output.size(), 1e-5);
    }
    
    PASSLOG();
}

int main(int argc, char** argv) {
    test_MLinear();
    return 0;
}