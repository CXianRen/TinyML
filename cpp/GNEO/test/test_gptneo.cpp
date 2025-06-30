#include "test_common.hpp"
#include "GPTNeo.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float


void test_GPTNeo(){
    START_TEST();
    GPTNeoModel<FP_T> model(4, 768);
    // Load the model state dict
    model.load_model("../model_state_dict");  
    
    Tensor<int> input_ids({1, 5}, {7454, 2402,257,640,11});
    Tensor<int> position_ids({1, 5}, {0, 1, 2, 3, 4});
    auto output = load_data<FP_T>(
        "build/test/temp/output.bin", 1* 5* 50257);

    Tensor<FP_T> output_t = model.forward(input_ids, position_ids);

    // Check the 
    compare_data<FP_T>(
        output_t.data().get(), output.data(), 
        output.size(), 1e-5);

    PASSLOG();
}

int main(int argc, char** argv) {
    test_GPTNeo();
    return 0;
}