#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include <numeric>

#include "tensor_def.hpp"
#include "mtb_bc.hpp"

namespace mtb {        
    // Constructor
    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape)
    {
        shape_ = shape;
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty");
        }
        // Calculate strides based on the shape
        strides_.resize(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape[i];
        }
        // Initialize data with the total size based on the shape
        size_t total_size = 1;
        for (size_t dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Shape dimensions must be positive");
            }
            total_size *= dim;
        }
        data_ = std::shared_ptr<T[]>(
            new T[total_size], 
            std::default_delete<T[]>());
        size_ = total_size;
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape, 
                const std::vector<T> &data)
        :Tensor(shape) {
        if (data.size() != size_) {
            throw std::invalid_argument(
                "Data size must match the total size of the shape");
        }
        // Copy data into the tensor
        memcpy(data_.get(), data.data(), size_ * sizeof(T));
    }

    // Copy constructor, shallow copy
    /*
        Tensor a;
        Tensor b = a; // copy assignment
       
        Tensor c = std::move(a); // move assignment

        Tensor d;
        d = b; // copy assignment
        d = std::move(c); // move assignment

    */
    template <typename T>
    Tensor<T>::Tensor(const Tensor &other)
        :shape_(other.shape_), 
        strides_(other.strides_), 
        data_(other.data_),
        size_(other.size_){}

    // Move constructor
    template <typename T>
    Tensor<T>::Tensor(Tensor &&other) noexcept
        :shape_(std::move(other.shape_)), 
        strides_(std::move(other.strides_)), 
        data_(std::move(other.data_)),
        size_(other.size_) {
        other.data_ = nullptr; 
        other.size_ = 0;
        other.strides_.clear();
        other.shape_.clear();
        // Nullify the moved-from object's data pointer
    }

    // Copy assignment operator, shallow copy
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T> &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        data_ = other.data_;
        size_ = other.size_;
        return *this;
    }

    // Move assignment operator
    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor<T> &&other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.data_ = nullptr; 
            other.size_ = 0;
            other.strides_.clear();
            other.shape_.clear();
            // Nullify the moved-from object's data pointer
        }
        return *this;
    }


    template <typename T>
    bool Tensor<T>::is_contiguous() const {
        // Check if the tensor is contiguous in memory
        if (shape_.empty()) return true; // Empty tensor is considered contiguous
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                std::cout << "Tensor is not contiguous: " 
                          << "expected stride " << expected_stride 
                          << ", got " << strides_[i] << std::endl;
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }
    
    // deep copy
    template <typename T>
    Tensor<T> Tensor<T>::copy() const {
        Tensor c(shape_);
        // c.data_ = std::shared_ptr<T[]>(
        //     new T[size_], 
        //     std::default_delete<T[]>());
        
        std::vector<size_t> index(shape_.size(), 0);
        for (size_t i = 0; i< size_; ++i){
            // Calculate the flat index
            size_t flat_index = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                flat_index += index[d] * strides_[d];
            }
            c.data_.get()[i] = data_.get()[flat_index];
            // update the index
            for (ssize_t d = shape_.size() - 1; d >= 0; --d) {
                if (++index[d] < shape_[d]) break;
                // Reset this dimension and carry over
                index[d] = 0; 
            }
        }
        return c;
    }
    

    // reshape
    template <typename T>
    Tensor<T>& Tensor<T>::reshape(const std::vector<size_t> &new_shape) {
        if (new_shape.empty()) {
            throw std::invalid_argument(
                "New shape cannot be empty");
        }
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            if (dim <= 0) {
                throw std::invalid_argument(
                    "Shape dimensions must be positive");
            }
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::invalid_argument(
                "New shape must have the same"\
                 "total size as the original shape");
        }
        shape_ = new_shape;
        // Update strides based on the new shape
        strides_.resize(new_shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= new_shape[i];
        }
        // No need to reallocate data_ since it remains the same
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::slice(
        const std::vector<std::pair<size_t, size_t>>& ranges) const{
        // Check if the ranges are valid
        if (ranges.empty()) {
            throw std::invalid_argument(
                "Ranges cannot be empty"); 
        }

        if (ranges.size() > shape_.size()) {
            throw std::invalid_argument(
                "Too many ranges for the tensor dimensions");
        }
        // Create a new shape based on the ranges
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].first >= shape_[i] ||
                ranges[i].second > shape_[i] ||
                ranges[i].first >= ranges[i].second) {
                throw std::out_of_range(
                    "Range out of bounds for dimension " +
                    std::to_string(i)); 
            }
            auto dim_size = ranges[i].second - ranges[i].first;
            new_shape.push_back(dim_size);
        }

        // shallow copy the data pointer
        Tensor<T> result = *this; // shallow copy
        result.shape_ = new_shape;
        
        // update the data pointer to point to the correct location
        size_t offset = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            offset += ranges[i].first * strides_[i];
        }
        
        result.data_ = std::shared_ptr<T[]>(data_,
            data_.get() + offset);

        // if the shape is 1, we can skip this dimension
        std::vector<size_t> new_strides;
        size_t new_size = 1;
        new_shape.clear();
        for (size_t i = 0; i < result.shape_.size(); ++i) {
            auto dim_size = result.shape_[i];
            new_size *= dim_size;
            if (result.shape_[i] == 1) {
                // Skip this dimension
                continue;
            }
            new_strides.push_back(result.strides_[i]);
            new_shape.push_back(result.shape_[i]);
        }
        // Update the size of the result tensor
        result.size_ = new_size;
        result.strides_ = new_strides;
        result.shape_ = new_shape;
        
        return result;
    }

    /* Op */
    // using [] to access a innner dimension tensor
    template <typename T>
    Tensor<T> Tensor<T>::operator[](const size_t i) const {
        if (i < 0 || i >= shape_[0]) {
            throw std::out_of_range("Index out of range");
        }
        // create a new tensor with the same data pointer
        Tensor<T> result= *this; // shallow copy
        
        // update the shape and strides of the result tensor
        result.shape_.erase(result.shape_.begin());
        size_t stride = result.strides_[0];
        result.strides_.erase(result.strides_.begin());

        // update the data pointer to point to the correct location
        result.data_ = std::shared_ptr<T[]>(
            data_, data_.get() + i * stride);
        
        // update the size of the result tensor
        size_t new_size = 1;
        for (size_t dim : result.shape_) {
            new_size *= dim;
        }
        result.size_ = new_size;

        return result;
    }

    // using (i) to access 1 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const size_t i) const {
        return data_.get()[i * strides_[0]];
    }

    // using (i, j) to access 2 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const size_t i, const size_t j) const {
        return data_.get()[i * strides_[0] + j * strides_[1]];
    }

    // using (i, j, k) to access 3 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(
        const size_t i, const size_t&  j,const size_t k) const {
        return data_.get()[i * strides_[0] + 
                            j * strides_[1] + 
                            k * strides_[2]];
    }

    // using (i, j, k, l) to access 4 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const size_t i, const size_t j, 
                    const size_t k, const size_t l) const {
        return data_.get()[i * strides_[0] + 
                            j * strides_[1] + 
                            k * strides_[2] + 
                            l * strides_[3]];
    }
    
    // math operations
    // Helper for elementwise operations
    template <typename T, typename Op>
    void elementwiseOp(
        const Tensor<T> &x, 
        const Tensor<T> &y,
        const Tensor<T> &r, 
        Op op, 
        const char* opname) 
    {
        // shallow copy
        auto a = x;
        auto b = y;

        if (a.shape() != b.shape()) {
            // try to broadcast the shapes
            auto new_shape = compute_broadcast_shape(
                a.shape(), b.shape());
            a = broadcast(a, new_shape);
            b = broadcast(b, new_shape);
        }

        if (a.shape() != b.shape()) {
            throw std::invalid_argument(
                std::string("Shapes must match for ") + opname);
        }

        // a general version for all cases
        const size_t ndim = a.shape().size();
        const size_t total = std::accumulate(
            a.shape().begin(), a.shape().end(), 1, 
            std::multiplies<>());
        
        // Compute strides for iteration
        const auto& a_shape = a.shape();
        const auto& b_shape = b.shape();
        const auto& a_strides = a.strides();
        const auto& b_strides = b.strides();

        auto a_ptr = a.data().get();
        auto b_ptr = b.data().get();    
        auto r_ptr = r.data().get();

        std::vector<size_t> indexes(ndim, 0);
        for (size_t idx = 0; idx < total; ++idx) {
            // Compute flat index with broadcasting
            size_t a_offset = 0;
            size_t b_offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                a_offset += (a_shape[d] == 1 ? 0 : indexes[d]) * a_strides[d];
                b_offset += (b_shape[d] == 1 ? 0 : indexes[d]) * b_strides[d];
            }

            // Apply the operation
            r_ptr[a_offset] = op(a_ptr[a_offset], b_ptr[b_offset]);

            // Advance multi-dimensional index
            for (ssize_t d = ndim - 1; d >= 0; --d) {
                indexes[d]++;
                if (indexes[d] < a_shape[d]) {
                    break;
                }
                indexes[d] = 0;
            }
        }
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
        // new tensor to hold the result
        Tensor<T> result(shape_);
        // Call the elementwise operation
        elementwiseOp(*this, other, result,
            std::plus<T>(), "addition");
        // Return the result tensor
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const T value) const {
        // Create a tensor from the scalar value
        Tensor<T> v_t = scalar_to_tensor(value);
        Tensor<T> result(shape_);
        // Call the elementwise operation
        elementwiseOp(*this, v_t, result,
            std::plus<T>(), "addition");
        // Return the result tensor
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
        Tensor<T> result(shape_);
        elementwiseOp(*this, other, result,
            std::minus<T>(), "subtraction");
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const T value) const {
        Tensor<T> v_t = scalar_to_tensor(value);
        Tensor<T> result(shape_);
        elementwiseOp(*this, v_t, result,
            std::minus<T>(), "subtraction");
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
        Tensor<T> result(shape_);
        elementwiseOp(*this, other, result,
            std::multiplies<T>(), "multiplication");
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const T value) const {
        Tensor<T> v_t = scalar_to_tensor(value);
        Tensor<T> result(shape_);
        elementwiseOp(*this, v_t, result,
            std::multiplies<T>(), "multiplication");
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
        Tensor<T> result(shape_);
        elementwiseOp(*this, other, result, 
            [](const T& a, const T& b) {
                if (b == 0) throw std::runtime_error("Division by zero");
                return a / b;
            }, "division");
        return result;
    }
    
    template <typename T>
    Tensor<T> Tensor<T>::operator/(const T value) const {
        if (value == 0) {
            throw std::runtime_error("Division by zero");
        }
        // Create a tensor from the scalar value
        Tensor<T> v_t = scalar_to_tensor(value);
        Tensor<T> result(shape_);
        // Call the elementwise operation
        elementwiseOp(*this, v_t, result,
            [](const T& a, const T& b) { return a / b; }, "division");
        return result;
    }

    // Inplace operations
    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T> &other) {
        elementwiseOp(
            *this, other, *this,
            std::plus<T>(), "addition");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const T value) {
        auto v_t = scalar_to_tensor(value);
        elementwiseOp(
            *this, v_t, *this,
            std::plus<T>(), "addition");
        return *this;
    }
    
    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T> &other) {
        elementwiseOp(
            *this, other, *this,
            std::minus<T>(), "subtraction");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const T value) {
        auto v_t = scalar_to_tensor(value);
        elementwiseOp(
            *this, v_t, *this,
            std::minus<T>(), "subtraction");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T> &other) {
        elementwiseOp(
            *this, other, *this,
            std::multiplies<T>(), "multiplication");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const T value) {
        auto v_t = scalar_to_tensor(value);
        elementwiseOp(
            *this, v_t, *this,
            std::multiplies<T>(), "multiplication");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T> &other) {
        // Custom lambda to check division by zero
        elementwiseOp(
            *this, other, *this,
            [](const T& a, const T& b) {
                if (b == 0) throw std::runtime_error("Division by zero");
                return a / b;
            }, "division");
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator/=(const T value) {
        if (value == 0) {
            throw std::runtime_error("Division by zero");
        }
        auto v_t = scalar_to_tensor(value);
        elementwiseOp(
            *this, v_t, *this,
            [](const T& a, const T& b) { return a / b; }, 
            "division");
        return *this;
    }
} // namespace mtb
