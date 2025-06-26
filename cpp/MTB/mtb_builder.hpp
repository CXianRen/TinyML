#pragma once

#include "tensor.hpp"
#include <vector>
#include <stdexcept>

namespace mtb {

// Create tensors
template <typename T>
Tensor<T> zeros(std::vector<size_t> shape) {
    Tensor<T> tensor(shape);
    // Initialize all elements to zero
    std::fill(tensor.data().get(),  
            tensor.data().get() + tensor.size(), 
            T(0));
    return tensor;
}

template <typename T>
Tensor<T> ones(std::vector<size_t> shape) {
    Tensor<T> tensor(shape);
    // Initialize all elements to one
    std::fill(tensor.data().get(), 
            tensor.data().get() + tensor.size(), 
            T(1));
    return tensor;
}

template <typename T>
Tensor<T> random(std::vector<size_t> shape, 
                 T min = T(0), T max = T(1)) {
    Tensor<T> tensor(shape);
    // Initialize with random values in the range [min, max)
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.data().get()[i] = 
        min + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max - min)));
    }
    return tensor;
}

template <typename T>
Tensor<T> triu(const Tensor<T> &tensor, int k = 0) {
    if (tensor.shape().size() != 2) {
        throw std::invalid_argument("Tensor must be 2D for triu operation");
    }
    Tensor<T> result(tensor.shape());

    auto s_ptr = tensor.data().get();
    auto d_ptr = result.data().get();

    size_t M = tensor.shape()[0];
    size_t N = tensor.shape()[1];

    size_t s0 = tensor.strides()[0];
    size_t s1 = tensor.strides()[1];
    size_t r0 = result.strides()[0];
    size_t r1 = result.strides()[1];

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // if k is positive
            if (j >= static_cast<int>(i) + k) {
                d_ptr[i * r0 + j * r1] = s_ptr[i * s0 + j * s1];
            } else {
                d_ptr[i * r0 + j * r1] = T(0);
            }
        }
    }
    return result;
}

} // namespace mtb