#include "test_common.hpp"
#include "GPTNeo.hpp"
#include "tokenizer.hpp"

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_generation(){
    GPTNeoModel<FP_T> model(4, 768);
    // Load the model state dict
    std::cout << "Loading model state dict from ../model_state_dict" << std::endl;

    model.load_model("../model_state_dict");  
    Tokenizer tokenizer("../python/tokenizer/vocab.json");
    tokenizer.parse_vocab();
    std::vector<int> input_ids_raw({7454,2402,257,640,11});
    std::vector<int> position_ids_raw( {0,1,2,3,4});
    size_t max_length = 30;
    
    //print the input tokens
    std::cout << "Input tokens:" << std::endl;
    for (const auto& token_id : input_ids_raw) {
        std::cout << tokenizer.get_token(token_id) << std::flush;
    }
    std::cout << std::endl;
    std::cout << "Generating tokens:" << std::endl;

    for (size_t i = 5; i < max_length; ++i) {
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
    std::cout << "Generation completed. max length: " 
    << max_length << std::endl;
}

int main(int argc, char** argv) {
    test_generation();
    return 0;
}