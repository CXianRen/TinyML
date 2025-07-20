#include "test_common.hpp"
#include "GPTNeo.hpp"
#include "tokenizer.hpp"
#include <chrono>
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
    size_t max_length = 105;
    
    //print the input tokens
    std::cout << "Input tokens:" << std::endl;
    for (const auto& token_id : input_ids_raw) {
        std::cout << tokenizer.get_token(token_id) << std::flush;
    }
    std::cout << std::endl;
    std::cout << "Generating tokens:" << std::endl;

    std::vector<Tensor<FP_T>> k_history = {
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
    };
    std::vector<Tensor<FP_T>> v_history = {
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
        Tensor<FP_T>({1}),
    };

    // measure the time taken for generation
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 5; i < max_length; ++i) {
        Tensor<int> input_ids({1, input_ids_raw.size()}, input_ids_raw);
        Tensor<int> position_ids({1, position_ids_raw.size()}, position_ids_raw);

        Tensor<FP_T> output_t = model.forward(
            input_ids, position_ids, &k_history, &v_history);
        
        auto next_token = output_t.slice(
            {{0,1}, 
            {input_ids_raw.size()-1,input_ids_raw.size()}, 
            {0,50257}});
  
        auto next_token_index = mtb::argmax(next_token, -1);

        auto next_token_str = tokenizer.get_token(
            next_token_index.data()[0]
        );
        // Append the next token to input_ids
        input_ids_raw.clear();
        input_ids_raw.push_back(next_token_index.data()[0]);
        position_ids_raw.clear();
        position_ids_raw.push_back(i);
        std::cout << next_token_str << std::flush;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << std::endl;
    std::cout << "\n Generation completed. max length: " 
    << max_length << std::endl;
    std::cout << "Time taken for generation: " 
              << 100.f /elapsed.count() << " token per second." << std::endl;

}

int main(int argc, char** argv) {
    test_generation();
    return 0;
}