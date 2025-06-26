#ifndef __MEmbed_HPP__
#define __MEmbed_HPP__
#include "mtb.hpp"

namespace mnn {

template <typename T>
class MEmbed {
    public:
        MEmbed(int vocab_size, int embed_dim): 
            vocab_size_(vocab_size), 
            embed_dim_(embed_dim),
            weight_(mtb::Tensor<T>({vocab_size, embed_dim})) {
        } 

        // Forward pass method
        mtb::Tensor<T> forward(const mtb::Tensor<int>& input) {
            // input should be a 1D tensor of indices
            // [B, S] where B is batch size and S is sequence length
            if (input.shape().size() != 2) {
                throw std::runtime_error(
                    "Input tensor must be 1D for embedding.");
            }

            std::vector<mtb::Tensor<T>> output_tensors;
            for (int i = 0; i < input.shape()[0]; ++i) {
                for (int j = 0; j < input.shape()[1]; ++j) {
                    auto index = input(i, j);
                    if (index < 0 || index >= vocab_size_) {
                        throw std::runtime_error(
                            "Index out of bounds for embedding.");
                    }
                    auto sub_tensor = weight_[index];
                    std::cout << "sub_tensor shape: " 
                              << sub_tensor.shape() << std::endl;
                    output_tensors.push_back(sub_tensor);
                }
            }
            
            // Concatenate the output tensors
            return mtb::concatenate(output_tensors);
        }

        void fill_weight(T* data, int size) {
            // Fill weight tensor with provided data
            if (size != weight_.size()) {
                throw std::runtime_error(
                    "Size mismatch for weight tensor.");
            }
            memcpy(weight_.data().get(), data, 
                   weight_.size() * sizeof(T));
        }
      
    private:
        int vocab_size_;
        int embed_dim_;
        // Weight and bias tensors
        mtb::Tensor<T> weight_;
};

} // namespace mnn

#endif // __MEmbed_HPP__
