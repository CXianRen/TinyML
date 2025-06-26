#ifndef __MNorm_HPP__
#define __MNorm_HPP__
#include "MTB.hpp"

namespace mnn {

template <typename T>
class MLayerNorm {
    public:
        MLayerNorm(int normalized_shape, float eps = 1e-5): 
            eps_(eps), normalized_shape_(normalized_shape) {
        } 

        // Forward pass method
        // mtb::Tensor<T> forward(const mtb::Tensor<T>& input) {
           
        // }
        
    private:
        double eps_;
        int normalized_shape_;
       
};

} // namespace mnn

#endif // __MNorm_HPP__
