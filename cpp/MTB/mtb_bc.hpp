#pragma once

#include "tensor.hpp"
namespace mtb {

// auto broadcast function
template <typename T>
std::vector<size_t> compute_broadcast_shape(
    const T &scalar,
    const std::vector<size_t> &shape) {
    return shape; // scalar can be broadcasted to any shape
}

std::vector<size_t> compute_broadcast_shape(
    const std::vector<size_t> &shape1,
    const std::vector<size_t> &shape2) {

    auto max_size = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result_shape(max_size, 1);
    for (size_t i = 0; i < max_size; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::invalid_argument(
                "Shapes are not compatible for broadcasting");
        }
        result_shape[max_size - 1 - i] = std::max(dim1, dim2);
    }
    return result_shape;
}


template <typename T>
Tensor<T> scalar_to_tensor(T value) {
    // Create a 0D tensor (scalar)
    std::vector<size_t> shape = {1};
    Tensor<T> tensor(shape);
    // Assign the value to the tensor
    tensor(0) = value;
    return tensor;
}

template <typename T>
Tensor<T> broadcast(const Tensor<T> &tensor, 
                 const std::vector<size_t> &shape) {
    // Check if the tensor can be broadcasted to the new shape
    if (tensor.shape().size() > shape.size()) {
        throw std::invalid_argument(
            "Tensor cannot be broadcasted to the new shape");
    }

    // create a shallow copy, for a new view.
    Tensor<T> result = tensor;
    
    /*
    will check if the tensor can be broadcasted to the new shape
    in other functions, when computing the parameter: shape
    here we just create a new view of the tensor
    with the new shape, and the same data pointer.
    and also update the strides
    */

    /*
       extend the dimensions to match the new shape.
       e.g. t = [2,3] new t = [1, 4, 2, 3]
            s = [3,1]
         t-> [1, 1, 2, 3] (padding 2 dimensions)
         s-> [0, 0, 3, 1] (padding 2 dimensions)

    */
    std::vector<size_t> r_shape = result.shape();
    std::vector<size_t> r_strides = result.strides();
    
    size_t padding = shape.size() - r_shape.size();
    for (size_t i = 0; i < padding; ++i) {
        r_shape.insert(r_shape.begin(), 1); // prepend 1
        r_strides.insert(r_strides.begin(), 0); // prepend 0 strides
    }

    // broadcast each dimension
    for (size_t i = 0; i < r_shape.size(); ++i) {
        if (r_shape[i] != shape[i])
        {
            // if the tensor shape is 1, then we can broadcast it
            r_shape[i] = shape[i];
            r_strides[i] = 0; // reset strides for the new dimension
        }
    }
    // update the result tensor shape and strides
    result.shape(r_shape);
    result.strides(r_strides);

    return result;
}

template <typename T>
Tensor<T> broadcast(const T &scalar, 
                    const std::vector<size_t> &shape){
    // Convert scalar to tensor
    Tensor<T> tensor = scalar_to_tensor(scalar);
    // Broadcast the tensor to the new shape
    return broadcast(tensor, shape);
}
}