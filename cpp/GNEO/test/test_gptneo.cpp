#include "test_common.hpp"
#include "GPTNeo.hpp"
#include "tokenizer.hpp"

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

    auto next_token = output_t.slice(
        {{0,1}, {4,5}, {0,50257}});
    
    std::cout << "shape of next token: "
              << next_token.shape() << 
              " size: " << next_token.size() 
              << " stride: " << next_token.strides()
              << std::endl;

    auto next_token_index = mtb::argmax(next_token, -1);
    std::cout << "next token index shape: "
              << next_token_index.shape() << std::endl;
    
    assert(next_token_index.size() == 1);
    assert(next_token_index.data()[0] == 612);

    Tokenizer tokenizer("../python/tokenizer/vocab.json");
    tokenizer.parse_vocab();

    auto next_token_str = tokenizer.get_token(
        next_token_index.data()[0]
    );
    std::cout << "next token: " << next_token_str << std::endl;

    PASSLOG();
}

void test_generation(){
    START_TEST();
    GPTNeoModel<FP_T> model(4, 768);
    // Load the model state dict
    model.load_model("../model_state_dict");  
    Tokenizer tokenizer("../python/tokenizer/vocab.json");
    tokenizer.parse_vocab();
    std::vector<int> input_ids_raw({7454,2402,257,640,11});
    std::vector<int> position_ids_raw( {0,1,2,3,4});
    int max_length = 30;
    for (int i = 5; i < max_length; ++i) {
        Tensor<int> input_ids({1, i}, input_ids_raw);
        Tensor<int> position_ids({1, i}, position_ids_raw);

        Tensor<FP_T> output_t = model.forward(
            input_ids, position_ids);

        auto next_token = output_t.slice(
            {{0,1}, 
            {i-1,i}, 
            {0,50257}});
  
        auto next_token_index = mtb::argmax(next_token, -1);

        auto next_token_str = tokenizer.get_token(
            next_token_index.data()[0]
        );
        // Append the next token to input_ids
        input_ids_raw.push_back(next_token_index.data()[0]);
        position_ids_raw.push_back(i);
        std::cout << next_token_str << std::flush;
    }
    std::cout << std::endl;

    PASSLOG();
}

int main(int argc, char** argv) {
    test_GPTNeo();
    test_generation();
    return 0;
}