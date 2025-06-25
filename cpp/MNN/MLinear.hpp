#ifndef __MLinear_HPP__
#define __MLinear_HPP__
#include "MTB.hpp"

namespace MNN {

template <typename T>
class MLinear {
    public:
        MLinear(int in_features, int out_features, bool bias = true):
            in_features_(in_features),
            out_features_(out_features),
            has_bias_(bias) {
            // Initialize weight tensor with random values
            weight_ = MTB::Tensor<T>({out_features_, in_features_}); 
            if (has_bias_) {
                // Initialize bias tensor with random values
                bias_ = MTB::Tensor<T>(
                    {out_features_});
            }
        }

        // Forward pass method
        MTB::Tensor<T> forward(const MTB::Tensor<T>& input) {
            // Perform matrix multiplication
            MTB::Tensor<T> output = 
                input.matmul(weight_.transpose(1, 0));

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
            memcpy(weight_.data(), data, 
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
            memcpy(bias_.data(), data, 
                   bias_.size() * sizeof(T));
        }
        
    private:
        bool has_bias_ = true;
        int in_features_;
        int out_features_;

        // Weight and bias tensors
        MTB::Tensor<T> weight_;
        MTB::Tensor<T> bias_;
};

} // namespace MNN

#endif // __MLinear_HPP__
