#pragma once

#include "tensor.hpp"
#include "mtb_bc.hpp"
#include "mtb_builder.hpp"
namespace mtb {

// transpose function
template <typename T>
Tensor<T> transpose(Tensor<T> &tensor, std::vector<size_t> axes) {
    // just update the shape and strides of the tensor

    // if axes is not equal to shape size, then throw error
    if (axes.size() != tensor.shape().size()) {
        throw std::invalid_argument("Axes size must match tensor shape size");
    }
    //
    std::vector<size_t> new_shape(tensor.shape().size());
    std::vector<size_t> new_strides(tensor.strides().size());
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
    size_t total_size = 0;
    std::vector<size_t> shape = tensors[0].shape();
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
    std::vector<size_t> new_shape = shape;
    // set the first dimension to the total size
    new_shape[0] = total_size; 
    Tensor<T> result(new_shape);
    // copy the data from each tensor into the result tensor
    size_t offset = 0;
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
    for (size_t i = 0; i < condition.size(); ++i) {
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
    std::vector<size_t> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1; 
    Tensor<T> result(new_shape);

    size_t last_dim = tensor.shape().back();

    size_t offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (size_t i = 0; i < tensor.size() / last_dim; ++i) {
        T max_val = t_ptr[offset];
        for (size_t j = 1; j < last_dim; ++j) {
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
    if (axes == static_cast<int>(tensor.shape().size()) - 1) {
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
    std::vector<size_t> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;

    Tensor<T> result(new_shape);
    
    // sum along the last dimension
    size_t last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    size_t offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (size_t i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        for (size_t j = 0; j < last_dim; ++j) {
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
    if (axes == static_cast<int>(tensor.shape().size()) - 1) {
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
    std::vector<size_t> new_shape = tensor.shape();

    if (tensor.strides().back() != 1) {
        throw std::invalid_argument(
            "Tensor must be contiguous for mean");
    }

    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;

    Tensor<T> result(new_shape);
    
    // compute the mean along the last dimension
    size_t last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    size_t offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (size_t i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        for (size_t j = 0; j < last_dim; ++j) {
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
    if (axes == static_cast<int>(tensor.shape().size()) - 1) {
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

    if (tensor.strides().back() != 1) {
        throw std::invalid_argument(
            "Tensor must be contiguous for variance");
    }

    // create a new tensor with the shape of the original tensor
    // except for the last dimension
    std::vector<size_t> new_shape = tensor.shape();
    // reduce last dimension to 1
    new_shape[new_shape.size() - 1] = 1;
    
    Tensor<T> result(new_shape);
    
    // compute the variance along the last dimension
    size_t last_dim = tensor.shape().back();
    
    // compute the offset for the last dimension
    size_t offset = 0;
    auto t_ptr = tensor.data().get();
    auto r_ptr = result.data().get();
    for (size_t i = 0; i < tensor.size() / last_dim; ++i) {
        T sum = 0;
        T mean = 0;
        for (size_t j = 0; j < last_dim; ++j) {
            sum += t_ptr[offset + j];
        }
        mean = sum / last_dim;
        
        T var_sum = 0;
        for (size_t j = 0; j < last_dim; ++j) {
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
    if (axes == static_cast<int>(tensor.shape().size()) - 1) {
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
    for (size_t i = 0; i < tensor.size(); ++i) {
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

template <typename T>
inline void softmax_1D(T* src, T*dst, size_t size, size_t stride = 1) {
    T max_val = src[0];
    for (size_t i = stride; i < size * stride; i += stride) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    T sum = 0;
    for (size_t i = 0; i < size * stride; i += stride) {
        dst[i] = std::exp(src[i] - max_val);
        sum += dst[i];
    }

    for (size_t i = 0; i < size * stride; i += stride) {
        dst[i] /= sum;
    }
}

template <typename T>
Tensor<T> softmax_2D(const Tensor<T> &tensor) {
    if (tensor.shape().size() != 2) {
        throw std::invalid_argument("Tensor must be 2D for softmax");
    }
    size_t rows = tensor.shape()[0];
    size_t cols = tensor.shape()[1];

    auto rows_stride = tensor.strides()[0];
    auto cols_stride = tensor.strides()[1];
    
    Tensor<T> result(tensor.shape());
    
    auto src_ptr = tensor.data().get();
    auto dst_ptr = result.data().get();
    
    for (size_t i = 0; i < rows; ++i) {
        softmax_1D(src_ptr + i * rows_stride, 
                   dst_ptr + i * rows_stride, 
                   cols, cols_stride);
    }
    return result;
}

template <typename T>
Tensor<T> softmax(const Tensor<T> &tensor, 
    int axis = -1){
        // softmax along the specified axis
    if (axis < -static_cast<int>(tensor.shape().size()) ||
        axis >= static_cast<int>(tensor.shape().size())) {
        throw std::invalid_argument("Axis out of range for softmax");   
    }

    if (axis < 0) {
        axis += tensor.shape().size(); // convert negative axis to positive
    }

    // if the last dimension is the axis, we can use 2D softmax
    if (axis == static_cast<int>(tensor.shape().size()) - 1) {
    // reshape the tensor to 2D if it is not already
        auto r_t = tensor; //shallow copy
        if (tensor.shape().size() == 1) {
            // if it is 1D, we can treat it as a 2D tensor
            r_t = r_t.reshape({1, r_t.shape()[0]});
        }
        // if more than 2D, we can reshape it to 2D, keep the last dimensions
        else if (r_t.shape().size() > 2) {
            auto last_dim = r_t.shape().back();
            std::vector<size_t> new_shape = 
                {r_t.size() / last_dim, last_dim};
            r_t = r_t.reshape(new_shape);
        }
        auto result = softmax_2D(r_t);
        // convert back to original shape if needed
        auto original_shape = tensor.shape();
        if (result.shape() != original_shape) {
            result = result.reshape(original_shape);
        }
        return result;
    } else {
        throw std::invalid_argument("Softmax only supports last dimension for now");
    }
} // softmax


template <typename T>
Tensor<int> argmax(const Tensor<T> &tensor, 
    int axis = -1) {
    const auto &shape = tensor.shape();
    const auto &strides = tensor.strides();

    int ndims = static_cast<int>(shape.size());
    if (axis < -ndims || axis >= ndims) {
        throw std::invalid_argument("Axis out of range for argmax");
    }
    if (axis < 0) {
        axis += ndims; // convert negative axis to positive
    }

    // create a result shape with axis set to 1
    std::vector<size_t> result_shape = shape;
    result_shape[axis] = 1;
    Tensor<int> result(result_shape);

    const T* src_ptr = tensor.data().get();
    int* dst_ptr = result.data().get();
    const auto& result_stride = result.strides();

    size_t outer_count = 1;
    for (int i = 0; i < axis; ++i) {
        if(i != 0) {
            outer_count *= shape[i];
        }
    }

    std::vector<size_t> coord(ndims, 0);
    for (size_t idx = 0; idx < outer_count; ++idx){
        // convert idx to coordinates
        size_t tmp = idx;
        for (int i = ndims - 1; i >= 0; --i) {
            if (i == axis){
                coord[i] = 0;
                // skip the axis we are reducing
                continue; 
            }
            coord[i] = tmp % shape[i];
            tmp /= shape[i];
        }

        // inner loop to find the max index
        size_t max_index = 0;
        T max_value;
        for (size_t j = 0; j < shape[axis]; ++j) {
            coord[axis] = j; // set the current axis
            size_t flat_index = 0;
            for (int i = 0; i < ndims; ++i) {
                flat_index += coord[i] * strides[i];
            }
            if (j == 0 || src_ptr[flat_index] > max_value) {
                max_value = src_ptr[flat_index];
                max_index = j;
            }
        }

        // set the result at the current outer index
        coord[axis] = 0; // set the max index
        size_t result_flat_index = 0;
        for (int i = 0; i < ndims; ++i) {
            result_flat_index += coord[i] * result_stride[i];
        }
        dst_ptr[result_flat_index] = static_cast<int>(max_index);
    }

    return result; 
} // argmax

} // namespace mtb