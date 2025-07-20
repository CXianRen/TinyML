#pragma once

#include "tensor.hpp"
#include "mtb_bc.hpp"
#include "mtb_builder.hpp"
#include "mtb_op_unary.hpp"

#include <functional>
#include <limits>
#include <sstream>

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
void __copy(const Tensor<T> &src, T* dst){
    auto& shape = src.shape();
    auto& strides = src.strides();
    auto size = src.size();

    auto src_ptr = src.data().get();

    std::vector<size_t> indexes(shape.size(), 0);

    for(size_t i = 0; i < size; ++i) {
        // convert i to index
        size_t index = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            index += indexes[d] * strides[d];
        }
        // copy the data
        dst[i] = src_ptr[index];

        // update the index
        for (ssize_t d = shape.size() - 1; d >= 0; --d) {
            if (++indexes[d] < shape[d]) break; 
            // reset the index for this dimension
            indexes[d] = 0; 
        }
    }
}

template <typename T>
Tensor<T> concatenate(
    const std::vector<Tensor<T>> &tensors, 
    int axis = 0) {
    // check if all tensors have the same shape except for the concatenation axis
    if (tensors.empty()) {
        throw std::invalid_argument("Input tensors cannot be empty");
    }
    
    for (const auto &tensor : tensors) {
        if (tensor.shape().size() != tensors[0].shape().size()) {
            throw std::invalid_argument(
                "All tensors must have the same number of dimensions");
        }
        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            if (i != static_cast<size_t>(axis) && 
                tensor.shape()[i] != tensors[0].shape()[i]) {
                throw std::invalid_argument(
                    "All tensors must have the same shape"\
                    " except for the concatenation axis");
            }
        }
    }

    // calculate the new shape
    std::vector<size_t> new_shape = tensors[0].shape();
    size_t total_size = 0;
    for (const auto &tensor : tensors) {
        total_size += tensor.shape()[axis];
    }
    new_shape[axis] = total_size;

    // create a new tensor with the concatenated shape
    Tensor<T> result(new_shape);

    // e.g. [1, 2, 3], axis = 1 ndim_before_axis = 1 
    size_t ndim_before_axis = axis;

    size_t total_subtensor = 1;
    // calculate the total subtensor for this tensor
    for (size_t d = 0; d < ndim_before_axis; ++d) {
        total_subtensor *= new_shape[d];
    }
       
    // split the tensors into subtensors according to the axis
    // and store them in a vector in order to concatenate
    std::vector<Tensor<T>> subtensors;
    std::vector<size_t> indexes(ndim_before_axis, 0);

    for (size_t i = 0; i < total_subtensor; ++i) {  
        // get the current tensor
        for (size_t j = 0; j < tensors.size(); ++j) {
            auto tensor = tensors[j];
            for (size_t d = 0; d < ndim_before_axis; ++d) {
                // get the subtensor
                tensor=tensor[indexes[d]]; 
            }
            // push the subtensor to the vector
            subtensors.push_back(tensor);
        }
       
        // update the index
        for (ssize_t d = ndim_before_axis - 1; d >= 0; --d) {
            if (++indexes[d] < new_shape[d]) break; 
            // reset the index for this dimension
            indexes[d] = 0; 
        }
    }

    // copy the data from each subtensor into the result tensor
    size_t offset = 0;
    auto r_ptr = result.data().get();
    for (const auto &subtensor : subtensors) {
        auto size = subtensor.size();
        // copy the data
        __copy(subtensor, r_ptr + offset);
        offset += size;
    }
    return result;
}

// where function
template <typename T>
Tensor<T> where(const Tensor<uint8_t> &condition,
                const T &y, 
                const Tensor<T> &tensor){
    // deep copy of tensor making sure it is contiguous
    auto result = tensor.copy();

    // broadcast the tensor to the new shape
    // and make sure it is contiguous
    auto cond_t = broadcast(
        condition, 
        tensor.shape() 
    );
    
    auto& shape = result.shape();

    auto& r_strides = result.strides();
    auto& c_strides = cond_t.strides();

    auto r_ptr = result.data().get();
    auto c_ptr = cond_t.data().get();

    const size_t ndim = shape.size();
    const size_t total = std::accumulate(
        shape.begin(), 
        shape.end(), 
        1, 
        std::multiplies<size_t>()
    );

    std::vector<size_t> indexes(ndim, 0);

    for (size_t i = 0; i < total; ++i) {
        // convert i to index
        size_t r_index = 0;
        size_t c_index = 0;
        for (size_t d = 0; d < ndim; ++d) {
            r_index += indexes[d] * r_strides[d];
            c_index += indexes[d] * c_strides[d];
        }

        // check the condition
        if (c_ptr[c_index]) {
            // if condition is true, set the value to y
            r_ptr[r_index] = y;
        }
        
        // update the index
        for (ssize_t d = ndim - 1; d >= 0; --d) {
            if (++indexes[d] < shape[d]) break; 
            // reset the index for this dimension
            indexes[d] = 0; 
        }
    }
    return result;
}


template <typename T, typename R>
Tensor<R> __reduceOp(
    const Tensor<T>& tensor,
    int axis,
    std::function<void(R&, const T&, size_t)> reducer, 
    R init_val,
    std::function<R(R, size_t)> finalizer = nullptr
) {
    const auto& src_shape = tensor.shape();
    const auto& src_strides = tensor.strides();
    const T* src_ptr = tensor.data().get();

    // Validate axis
    int ndims = static_cast<int>(src_shape.size());
    if (axis < -ndims || axis >= ndims) {
        throw std::invalid_argument("Axis out of range");
    }

    if (axis < 0) axis += ndims;

    // Create a result shape with the specified axis set to 1
    std::vector<size_t> dst_shape = src_shape;
    dst_shape[axis] = 1;
    Tensor<R> dst(dst_shape);
    R* dst_ptr = dst.data().get();
    const auto& dst_strides = dst.strides();

    // Calculate the number of outer elements 
    // (elements not along the reduction axis)
    size_t outer_count = 1;
    for (int i = 0; i < ndims; ++i) {
        if (i != axis) outer_count *= src_shape[i];
    }

    std::vector<size_t> coord(ndims, 0);
    for (size_t idx = 0; idx < outer_count; ++idx) {
        size_t tmp = idx;
        for (int i = ndims - 1; i >= 0; --i) {
            if (i == axis) continue;
            coord[i] = tmp % src_shape[i];
            tmp /= src_shape[i];
        }

        // Initialize the reduction value
        R value = init_val;
        // Iterate over the elements along the reduction axis
        for (size_t j = 0; j < src_shape[axis]; ++j) {
            coord[axis] = j;
            size_t flat_index = 0;
            for (int i = 0; i < ndims; ++i)
                flat_index += coord[i] * src_strides[i];
            reducer(value, src_ptr[flat_index], j);
        }

        if (finalizer)
            value = finalizer(value, src_shape[axis]);

        coord[axis] = 0;
        size_t dst_flat_index = 0;
        for (int i = 0; i < ndims; ++i)
            dst_flat_index += coord[i] * dst_strides[i];
        dst_ptr[dst_flat_index] = value;
    }

    return dst;
}


template <typename T>
Tensor<T> max(const Tensor<T>& tensor, int axis = -1) {
    return __reduceOp<T, T>(
        tensor,
        axis,
        [](T& acc, const T& val, size_t j) {
            if (j == 0 || val > acc) acc = val;
        },
        std::numeric_limits<T>::lowest()
    );
}

// sum function
template <typename T>
Tensor<T> sum(const Tensor<T>& tensor, int axis = -1) {
    return __reduceOp<T, T>(
        tensor,
        axis,
        [](T& acc, const T& val, size_t) { acc += val; },
        static_cast<T>(0)
    );
}

// mean function
template <typename T>
Tensor<T> mean(const Tensor<T>& tensor, int axis = -1) {
    return __reduceOp<T, T>(
        tensor,
        axis,
        [](T& acc, const T& val, size_t) { acc += val; },
        static_cast<T>(0),
        [](T total, size_t count) { return total / static_cast<T>(count); }
    );
}

template <typename T>
Tensor<int> argmax(const Tensor<T>& tensor, int axis = -1) {
    return __reduceOp<T, int>(
        tensor,
        axis,
        [&, tensor](int& acc, const T& val, size_t j) {
            static T max_val;
            if (j == 0 || val > max_val) {
                max_val = val;
                acc = static_cast<int>(j);
            }
        },
        0
    );
}

// var function
template <typename T>
Tensor<T> var(const Tensor<T> &tensor, 
                int axis) {
    int ndims = static_cast<int>(tensor.shape().size());
    if (axis  < -ndims || axis >= ndims) {
        throw std::invalid_argument("Axis out of range for argmax");
    }

    if (axis < 0) {
        // convert negative axis to positive
        axis += ndims; 
    }

    auto m = mean<T>(tensor, axis);
    auto c = tensor - m;
    auto sqr = pow(c, 2);
    auto ret = mean(sqr, axis);
    return ret;
}

template <typename T>
Tensor<T> softmax(const Tensor<T> &tensor, 
    int axis = -1){
    // softmax along the specified axis
    int ndims = static_cast<int>(tensor.shape().size());
    if (axis  < -ndims || axis >= ndims) {
        std::stringstream ss;
        ss << "Axis out of range for argmax: " << axis 
           << ", ndims: " << ndims;
        throw std::invalid_argument(ss.str());
    }

    if (axis < 0) {
        // convert negative axis to positive
        axis += ndims; 
    }

    auto max_val = max(tensor, axis);
    auto shifted = tensor - max_val;
    auto exps = exp(shifted);
    auto sum_exps = sum(exps, axis);
    auto ret = exps / sum_exps;
    return ret;
}

} // namespace mtb