#ifndef __MLinear_HPP__
#define __MLinear_HPP__
#include "mtb.hpp"
#include "Mmodel.hpp"

namespace mnn {

template <typename T>
class MLinear: public MModel {
    public:
        MLinear(size_t in_features, 
                size_t out_features, 
                bool bias = true):
            has_bias_(bias),
            in_features_(in_features),
            out_features_(out_features),
            weight_({out_features_, in_features_}),
            bias_(has_bias_ ? mtb::Tensor<T>({out_features_}) 
            : mtb::Tensor<T>({1})){
            MACRO_CLASS_NAME(MLinear);
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

        void fill_weight(T* data, size_t size) {
            // Fill weight tensor with provided data
            if (size != weight_.size()) {
                throw std::runtime_error(
                    "Size mismatch for weight tensor.");
            }
            memcpy(weight_.data().get(), data, 
                   weight_.size() * sizeof(T));
        }

        void fill_bias(T* data, size_t size) {
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
      
        void printInfo(size_t indent = 0, 
            std::ostream& os = std::cout) const override {
            os << std::string(indent, ' ') 
                      << "(" << getModelName() << ") : " 
                      << getModelType() 
                      << " (in_features: " 
                      << in_features_ 
                      << ", out_features: " 
                      << out_features_  
                      << ", has_bias: "
                      << (has_bias_ ? "true" : "false")
                      <<
                      ")" << std::endl;
        }

        void loadParameters(
            const std::string& modelPath) override {
            auto data = load_data<T>(
                modelPath + "/weight/parameters.bin");
            fill_weight(data.data(), data.size());
            if (has_bias_) {
                auto bias_data = load_data<T>(
                    modelPath + "/bias/parameters.bin");
                fill_bias(bias_data.data(), bias_data.size());
            }
        }


    private:
        bool has_bias_ = true;
        size_t in_features_;
        size_t out_features_;

        // Weight and bias tensors
        mtb::Tensor<T> weight_;
        mtb::Tensor<T> bias_;
};

} // namespace mnn

#endif // __MLinear_HPP__
