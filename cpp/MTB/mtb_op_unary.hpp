#pragma once

#include "tensor.hpp"
#include "mtb_bc.hpp"
#include "mtb_builder.hpp"
namespace mtb {

template <typename T, typename F>
Tensor<T> unary_elementwise(const Tensor<T> &tensor, F func, const char* opname) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument(
            "Tensor must have at least one dimension");
    }
    Tensor<T> result(tensor.shape());
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (size_t i = 0; i < tensor.size(); ++i) {
        r_ptr[i] = func(t_ptr[i]);
    }
    return result;
}

// sqrt function
template <typename T>
Tensor<T> sqrt(const Tensor<T> &tensor) {
    return unary_elementwise<T>(tensor, [](const T& v) {
        if (v < 0) throw std::invalid_argument(
            "Cannot compute square root of negative number");
        return std::sqrt(v);
    }, "sqrt");
}

// pow function
template <typename T>
Tensor<T> pow(const Tensor<T> &tensor, int exponent) {
    return unary_elementwise<T>(tensor, [exponent](const T& v) {
        return std::pow(v, exponent);
    }, "pow");
}

// exp function
template <typename T>
Tensor<T> exp(const Tensor<T> &tensor) {
    return unary_elementwise<T>(tensor, [](const T& v) {
        return std::exp(v);
    }, "exp");
}

// tanh function
template <typename T>
Tensor<T> tanh(const Tensor<T> &tensor) {
    return unary_elementwise<T>(tensor, [](const T& v) {
        return std::tanh(v);
    }, "tanh");
}

} // namespace mtb