#ifndef __MLinear_HPP__
#define __MLinear_HPP__
#include "mtb.hpp"

namespace mnn {

template <typename T>
class MLinear {
    public:
        MLinear(int in_features, int out_features, bool bias = true):
            in_features_(in_features),
            out_features_(out_features),
            has_bias_(bias),
            weight_({out_features_, in_features_}),
            bias_(has_bias_ ? mtb::Tensor<T>({out_features_}) : mtb::Tensor<T>({1})){
        }

        // Forward pass method
        mtb::Tensor<T> forward(const mtb::Tensor<T>& input) {
            // Perform matrix multiplication
            mtb::Tensor<T> output = 
                mtb::matmul(input, mtb::transpose(weight_, {1, 0}));

            // Add bias if it exists
            if (has_bias_) {
                // bias will be broadcasted automatically
                output += bias_;
            }
            return output;  
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

        void fill_bias(T* data, int size) {
            // Fill bias tensor with provided data
            if (!has_bias_) {
                throw std::runtime_error(
                    "Bias is not enabled for this layer.");
            }
            if (size != bias_.size()) {
                throw std::runtime_error(
                    "Size mismatch for bias tensor.");
            }
            memcpy(bias_.data().get(), data, 
                   bias_.size() * sizeof(T));
        }
        
    private:
        bool has_bias_ = true;
        int in_features_;
        int out_features_;

        // Weight and bias tensors
        mtb::Tensor<T> weight_;
        mtb::Tensor<T> bias_;
};

} // namespace mnn

#endif // __MLinear_HPP__
