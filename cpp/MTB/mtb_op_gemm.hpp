#pragma once

#include "tensor.hpp"
#include "mtb_bc.hpp"
#include "mtb_op_math.hpp"
namespace mtb {

// matmul function 2D
template <typename T>
void _GEMM(const Tensor<T>& a, const Tensor<T>& b, T* data){
    // a: [M, N], b: [N, K] -> result: [M, K]
 
    size_t M = a.shape()[0];
    size_t N = a.shape()[1];
    size_t K = b.shape()[1];

    size_t a_stride0 = a.strides()[0];
    size_t a_stride1 = a.strides()[1];
    size_t b_stride0 = b.strides()[0];
    size_t b_stride1 = b.strides()[1];

    auto a_ptr = a.data().get();
    auto b_ptr = b.data().get();

    // Initialize the result tensor
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            for (size_t k = 0; k < N; ++k) {
                data[i * K + j] += 
                    a_ptr[i * a_stride0 + k * a_stride1] * 
                    b_ptr[k * b_stride0 + j * b_stride1];
            }
        }
    }
}

std::vector<std::vector<size_t>> 
generate_batch_indices(
    const std::vector<size_t>& shape) {
    std::vector<std::vector<size_t>> result;
    
    int ndim = static_cast<int>(shape.size());
    
    std::vector<size_t> indices(ndim, 0);
    while (true) {
        result.push_back(indices);
        int i = ndim - 1;
        while (i >= 0) {
            indices[i]++;
            if (indices[i] < shape[i]) {
                break;
            }
            indices[i] = 0;
            i--;
        }
        if (i < 0) {
            break; // all indices have been generated
        }
    }
    return result;
}

// matmul function
template <typename T>
Tensor<T> matmul(const Tensor<T>& m1, const Tensor<T>& m2) {
    Tensor<T> a = m1;
    Tensor<T> b = m2;

    // fist check if the shapes are not equal
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // check and broadcast the shapes
    auto max_shape = a_shape.size() > b_shape.size() ? a_shape : b_shape;
    auto new_a_shape = max_shape;
    auto new_b_shape = max_shape;
    // copy the shapes to the new shapes from the back
    for (size_t i = 0; i < a_shape.size(); i++){
        new_a_shape[max_shape.size() - 1 - i] = 
            a_shape[a_shape.size() - 1 - i];
    }
    for (size_t i = 0; i < b_shape.size(); i++){
        new_b_shape[max_shape.size() - 1 - i] = 
            b_shape[b_shape.size() - 1 - i];
    }
    a = broadcast(a, new_a_shape);
    b = broadcast(b, new_b_shape);


    // check if the last 2 dimensions of a and b 
    // compatible for matrix multiplication
    if (a_shape.size() < 2 || b_shape.size() < 2)
    {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matrix multiplication");
    }

    if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2]) {
        
        std::stringstream ss;
        ss << "Shapes are not compatible for matrix multiplication: "
           << "a: " << a_shape << ", b: " << b_shape;
        throw std::invalid_argument(ss.str());
    }
    
    // compute the shape of the result tensor
    // [x,x, M, N] @ [x,x, N, K] = [x,x, M, K]
    size_t M = a_shape[a_shape.size() - 2];
    // size_t N = a_shape[a_shape.size() - 1];
    size_t K = b_shape[b_shape.size() - 1];

    std::vector<size_t> result_shape = a_shape;
    result_shape[result_shape.size() - 2] = M;
    result_shape[result_shape.size() - 1] = K;

    Tensor<T> result = mtb::zeros<T>(result_shape);

    // generate batch shape
    std::vector<size_t> batch_shape =
        std::vector<size_t>(a_shape.begin(), 
                         a_shape.end() - 2);
    
    // generate indexes for batch dimensions
    auto batch_indices = generate_batch_indices(batch_shape);

    // Always use the batch loop, even if batch_indices is empty (no batch dims)
    for (size_t i = 0; 
        i < std::max(static_cast<size_t>(1), batch_indices.size()); 
        ++i) 
    {
        std::vector<size_t> indices;
        if (!batch_indices.empty()) {
            indices = batch_indices[i];
        }
        auto sub_a = a;
        auto sub_b = b;
        for (size_t j = 0; j < indices.size(); ++j) {
            sub_a = sub_a[indices[j]];
            sub_b = sub_b[indices[j]];
        }
        _GEMM(sub_a, sub_b, result.data().get() + i * M * K);
    }
    return result;
}

}