#ifndef __MNorm_HPP__
#define __MNorm_HPP__
#include "mtb.hpp"

namespace mnn {

template <typename T>
class MLayerNorm {
    public:
        MLayerNorm(size_t normalized_shape, double eps = 1e-5): 
            eps_(eps), normalized_shape_(normalized_shape),
            gamma_(mtb::ones<T>(
                {normalized_shape})),
            beta_(mtb::zeros<T>(
                {normalized_shape})) {}

        // Forward pass method
        mtb::Tensor<T> forward(const mtb::Tensor<T>& input) {
            auto mean = mtb::mean(input, -1);
            auto variance = mtb::var(input, -1);
            // Add epsilon to variance for numerical stability
            variance += eps_; 
            auto stddev = mtb::sqrt(variance);
            auto normalized = (input - mean);
            normalized /= stddev;
            normalized *= gamma_;
            normalized += beta_;
            return normalized;
        }
        
        void fill_gamma(T *data, size_t size) {
            if (size != gamma_.size()) {
                throw std::runtime_error("Size mismatch for gamma tensor");
            }
            memcpy(gamma_.data().get(), 
            data, size * sizeof(T));
        }

        void fill_beta(T *data, size_t size) {
            if (size != beta_.size()) {
                throw std::runtime_error("Size mismatch for beta tensor");
            }
            memcpy(beta_.data().get(), 
            data, size * sizeof(T));
        }

    private:
        double eps_;
        size_t normalized_shape_;
        
        mtb::Tensor<T> gamma_;
        mtb::Tensor<T> beta_;
};

} // namespace mnn

#endif // __MNorm_HPP__
