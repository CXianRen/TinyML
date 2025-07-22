#ifndef __MEmbed_HPP__
#define __MEmbed_HPP__
#include "mtb.hpp"
#include "Mmodel.hpp"
namespace mnn {

template <typename T>
class MEmbed: public MModel {
    public:
        MEmbed(size_t vocab_size, size_t embed_dim):
            vocab_size_(vocab_size), 
            embed_dim_(embed_dim),
            weight_(mtb::Tensor<T>({vocab_size, embed_dim})) {
            MACRO_CLASS_NAME(MEmbed); 
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
            for (size_t i = 0; i < input.shape()[0]; ++i) {
                for (size_t j = 0; j < input.shape()[1]; ++j) {
                    auto index = input(i, j);
                    if (index < 0 || 
                        static_cast<size_t>(index) >= vocab_size_) {
                        throw std::runtime_error(
                            "Index out of bounds for embedding.");
                    }
                    auto sub_tensor = weight_[index];
                    output_tensors.push_back(sub_tensor);
                }
            }
            
            // Concatenate the output tensors
            auto out = mtb::concatenate(output_tensors);
            // Reshape to [B, S, embed_dim]
            return out.reshape(
                {input.shape()[0], input.shape()[1], embed_dim_});
        }

        void fill_weight(T* data, size_t size) {
            // Fill weight tensor with provided data
            if (size != weight_.size()) {
                throw std::runtime_error(
                    "Size mismatch for weight tensor.");
            }
            memcpy(weight_.data().get(), data, 
                   weight_.size() * sizeof(T));
        }

        void printInfo(size_t indent = 0, 
            std::ostream& os = std::cout) const override {
            os << std::string(indent, ' ') 
                      << "(" << getModelName() << ") : "
                      << getModelType() <<
                      "(" << vocab_size_ 
                      << ", " << embed_dim_ << ")" 
                      << std::endl;
        }

        void loadParameters(const std::string& modelPath) override {
            auto data = load_data<T>(
                modelPath + "/weight/parameters.bin");
            fill_weight(data.data(), data.size());
        }

    private:
        size_t vocab_size_;
        size_t embed_dim_;
        // Weight and bias tensors
        mtb::Tensor<T> weight_;
};

} // namespace mnn

#endif // __MEmbed_HPP__
