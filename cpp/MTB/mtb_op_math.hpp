#pragma once

#include "tensor.hpp"
#include "mtb_bc.hpp"
#include "mtb_builder.hpp"
namespace mtb {

// transpose function
template <typename T>
Tensor<T> transpose(Tensor<T> &tensor, std::vector<int> axes) {
    // just update the shape and strides of the tensor

    // if axes is not equal to shape size, then throw error
    if (axes.size() != tensor.shape().size()) {
        throw std::invalid_argument("Axes size must match tensor shape size");
    }
    //
    std::vector<int> new_shape(tensor.shape().size());
    std::vector<int> new_strides(tensor.strides().size());
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < 0 || axes[i] >= tensor.shape().size()) {
            throw std::out_of_range("Axis index out of range");
        }
        new_shape[i] = tensor.shape()[axes[i]];
        new_strides[i] = tensor.strides()[axes[i]];
    }
    // update the tensor shape and strides
    auto new_tensor = tensor; // shallow copy
    new_tensor.shape(new_shape);
    new_tensor.strides(new_strides);
    return new_tensor;
}

// concatenate function
template <typename T>
Tensor<T> concatenate_dim0(const std::vector<Tensor<T>> &tensors) {
    // concatenate tensors along dimension 0
    // for fast concatenation,
    if (tensors.empty()) {
        throw std::invalid_argument("Input tensors cannot be empty");
    }

    // check if all tensors have the same shape except for the first dimension
    int total_size = 0;
    std::vector<int> shape = tensors[0].shape();
    for (const auto &tensor : tensors) {
        if (tensor.shape().size() != shape.size()) {
            throw std::invalid_argument(
                "All tensors must have the same number of dimensions");
        }
        for (size_t i = 1; i < shape.size(); ++i) {
            if (tensor.shape()[i] != shape[i]) {
                throw std::invalid_argument(
                    "All tensors must have the same shape"\
                    " except for the first dimension");
            }
        }
        total_size += tensor.shape()[0];
    }

    // create a new tensor with the concatenated shape
    std::vector<int> new_shape = shape;
    // set the first dimension to the total size
    new_shape[0] = total_size; 
    Tensor<T> result(new_shape);
    // copy the data from each tensor into the result tensor
    int offset = 0;
    for (const auto &tensor : tensors) {
        auto s_ptr = tensor.data().get();
        auto d_ptr = result.data().get();
        if(tensor.is_contiguous()){
             // copy the data
            memcpy(d_ptr + offset, 
                   s_ptr, 
                   tensor.size() * sizeof(T));
            offset += tensor.size();
        }else{
            // unspported case
            throw std::invalid_argument(
                "Tensor is not contiguous, cannot concatenate");
        }
    }
    return result;
}

template <typename T>
Tensor<T> concatenate(const std::vector<Tensor<T>> &tensors, 
    int axis = 0) {
    switch (axis) {
        case 0:
            return concatenate_dim0(tensors);
        default:
            throw std::invalid_argument("Unsupported axis for concatenation");
    }
}

// where function
template <typename T>
Tensor<T> where(const Tensor<bool> &condition, 
                 const Tensor<T> &tensor, 
                 T y) {
    if (condition.shape() != tensor.shape()) {
        throw std::invalid_argument(
            "Condition and tensor must have the same shape");
    }
    Tensor<T> result(condition.shape());
    auto r_ptr = result.data().get();
    auto t_ptr = tensor.data().get();
    auto cond_ptr = condition.data().get();
    for (int i = 0; i < condition.size(); ++i) {
        r_ptr[i] = cond_ptr[i] ? t_ptr[i] : y;
    }
    return result;
}

// max function
template <typename T>
Tensor<T> max_lastdim(const Tensor<T> &tensor){
    if (tensor.shape().empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    // create a new tensor with the shape of the original tensor
    // except for the last dimension
    std::vector<int> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1; 
    Tensor<T> result(new_shape);

    int last_dim = tensor.shape().back();

    int offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (int i = 0; i < tensor.size() / last_dim; ++i) {
        T max_val = t_ptr[offset];
        for (int j = 1; j < last_dim; ++j) {
            if (t_ptr[offset + j] > max_val) {
                max_val = t_ptr[offset + j];
            }
        }
        r_ptr[i] = max_val;
        offset += last_dim;
    }

    return result;
}

template <typename T>
Tensor<T> max(const Tensor<T> &tensor, 
                int axes) {
    // is the last dimension
    if (axes == tensor.shape().size() - 1) {
        axes = -1; // max along the last dimension
    }

    switch (axes) {
        case -1: // max along the last dimension
            return max_lastdim(tensor);
        default:
            throw std::invalid_argument(
                "Unsupported axes for max");
    }    
}

// sum function
template <typename T>
Tensor<T> sum_lastdim(const Tensor<T> &tensor) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    // create a new tensor with the shape of the original tensor
    // except for the last dimension
    std::vector<int> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;

    Tensor<T> result(new_shape);
    
    // sum along the last dimension
    int last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    int offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (int i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        for (int j = 0; j < last_dim; ++j) {
            sum += t_ptr[offset + j];
        }
        r_ptr[i] = sum;
        offset += last_dim;
    }
    return result;
}

template <typename T>
Tensor<T> sum(const Tensor<T> &tensor, 
                int axes) {
    // is the last dimension
    if (axes == tensor.shape().size() - 1) {
        axes = -1; // sum along the last dimension
    }

    switch (axes) {
        case -1: // sum along the last dimension
            return sum_lastdim(tensor);
        default:
            throw std::invalid_argument(
                "Unsupported axes for sum");
    }    
}

// mean function
template <typename T>
Tensor<T> mean_lastdim(const Tensor<T> &tensor) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    // create a new tensor with the shape of the original tensor
    // except for the last dimension
    std::vector<int> new_shape = tensor.shape();

    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;

    Tensor<T> result(new_shape);
    
    // compute the mean along the last dimension
    int last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    int offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (int i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        for (int j = 0; j < last_dim; ++j) {
            sum += t_ptr[offset + j];
        }
        r_ptr[i] = sum / last_dim;
        offset += last_dim;
    }
    return result;
}

template <typename T>
Tensor<T> mean(const Tensor<T> &tensor, 
                int axes) {
    // is the last dimension
    if (axes == tensor.shape().size() - 1) {
        axes = -1; // mean along the last dimension
    }

    switch (axes) {
        case -1: // mean along the last dimension
            return mean_lastdim(tensor);
        default:
            throw std::invalid_argument(
                "Unsupported axes for mean");
    }    
}

// var function
template <typename T>
Tensor<T> var_lastdim(const Tensor<T> &tensor) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    // create a new tensor with the shape of the original tensor
    // except for the last dimension
    std::vector<int> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;
    
    Tensor<T> result(new_shape);
    
    // compute the variance along the last dimension
    int last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    int offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (int i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        T mean = 0;
        for (int j = 0; j < last_dim; ++j) {
            sum += t_ptr[offset + j];
        }
        mean = sum / last_dim;
        
        T var_sum = 0;
        for (int j = 0; j < last_dim; ++j) {
            auto diff = t_ptr[offset + j] - mean;
            var_sum += diff * diff;
        }
        r_ptr[i] = var_sum / last_dim;
        offset += last_dim;
    }
    return result;
}

template <typename T>
Tensor<T> var(const Tensor<T> &tensor, 
                int axes) {
    // is the last dimension
    if (axes == tensor.shape().size() - 1) {
        axes = -1; // var along the last dimension
    }

    switch (axes) {
        case -1: // var along the last dimension
            return var_lastdim(tensor);
        default:
            throw std::invalid_argument(
                "Unsupported axes for var");
    }
}

template <typename T, typename F>
Tensor<T> unary_elementwise(const Tensor<T> &tensor, F func, const char* opname) {
    if (tensor.shape().empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    Tensor<T> result(tensor.shape());
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (int i = 0; i < tensor.size(); ++i) {
        r_ptr[i] = func(t_ptr[i]);
    }
    return result;
}

// sqrt function
template <typename T>
Tensor<T> sqrt(const Tensor<T> &tensor) {
    return unary_elementwise<T>(tensor, [](const T& v) {
        if (v < 0) throw std::invalid_argument("Cannot compute square root of negative number");
        return std::sqrt(v);
    }, "sqrt");
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

}