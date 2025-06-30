#ifndef __MActivation_HPP__
#define __MActivation_HPP__
#include "mtb.hpp"

namespace mnn {

template <typename T>
class MGELUActivation {
    public:
        MGELUActivation() = default;
        
        // Forward pass method
        mtb::Tensor<T> forward(const mtb::Tensor<T>& x) {
            // GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * 
            //  (x + 0.044715*x^3) ))

            // tanh(sqrt(2/pi) * (x + 0.044715*x^3))
            auto out = mtb::pow(x, 3);
            out *= 0.044715;
            out += x;
            out *= std::sqrt(2.0 / M_PI);
            out = mtb::tanh(out);
            out += 1.0; // 1 + tanh(...)
            
            out *= x;
            out *= 0.5; // 0.5 * x * (1 + tanh(...))
            return out; // GELU(x)
        }       
};

} // namespace mnn

#endif // __MActivation_HPP__
