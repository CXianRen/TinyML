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
    Tensor<T>::Tensor(const std::vector<int> &shape)
        :shape_(shape){
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty");
        }
        // Calculate strides based on the shape
        strides_.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape[i];
        }
        // Initialize data with the total size based on the shape
        int total_size = 1;
        for (int dim : shape) {
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
    Tensor<T>::Tensor(const std::vector<int> &shape, 
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
        size_(other.size_),
        data_(other.data_) {}

    // Move constructor
    template <typename T>
    Tensor<T>::Tensor(Tensor &&other) noexcept
        :shape_(std::move(other.shape_)), 
        strides_(std::move(other.strides_)), 
        size_(other.size_),
        data_(std::move(other.data_)) {
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
        int expected_stride = 1;
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
        c.data_ = std::shared_ptr<T[]>(
            new T[size_], 
            std::default_delete<T[]>());
        memcpy(c.data_.get(), data_.get(), size_ * sizeof(T));
        return c;
    }


    // reshape
    template <typename T>
    Tensor<T>& Tensor<T>::reshape(const std::vector<int> &new_shape) {
        if (new_shape.empty()) {
            throw std::invalid_argument(
                "New shape cannot be empty");
        }
        int new_size = 1;
        for (int dim : new_shape) {
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
        int stride = 1;
        for (int i = new_shape.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= new_shape[i];
        }
        // No need to reallocate data_ since it remains the same
        return *this;
    }


    /* Op */
    // using [] to access a innner dimension tensor
    template <typename T>
    Tensor<T> Tensor<T>::operator[](const int i) const {
        if (i < 0 || i >= shape_[0]) {
            throw std::out_of_range("Index out of range");
        }
        // create a new tensor with the same data pointer
        Tensor<T> result= *this; // shallow copy
        
        // update the shape and strides of the result tensor
        result.shape_.erase(result.shape_.begin());
        int stride = result.strides_[0];
        result.strides_.erase(result.strides_.begin());

        // update the data pointer to point to the correct location
        result.data_ = std::shared_ptr<T[]>(
            data_, data_.get() + i * stride);
        
        // update the size of the result tensor
        int new_size = 1;
        for (int dim : result.shape_) {
            new_size *= dim;
        }
        result.size_ = new_size;

        return result;
    }

    // using (i) to access 1 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const int i) const {
        return data_.get()[i * strides_[0]];
    }

    // using (i, j) to access 2 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const int i, const int j) const {
        return data_.get()[i * strides_[0] + j * strides_[1]];
    }

    // using (i, j, k) to access 3 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(
        const int i, const int&  j,const int k) const {
        return data_.get()[i * strides_[0] + 
                            j * strides_[1] + 
                            k * strides_[2]];
    }

    // using (i, j, k, l) to access 4 dimension tensor
    template <typename T>
    T& Tensor<T>::operator()(const int i, const int j, 
                    const int k, const int l) const {
        return data_.get()[i * strides_[0] + 
                            j * strides_[1] + 
                            k * strides_[2] + 
                            l * strides_[3]];
    }
    
    // math operations
    // Helper for elementwise operations
    template <typename T>
    template <typename Op>
    Tensor<T> Tensor<T>::elementwiseOp(
        const Tensor<T> &other, Op op, const char* opname) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(
                std::string("Shapes must match for ") + opname);
        }
        Tensor result(shape_);
        for (int i = 0; i < size_; ++i) {
            result.data_.get()[i] = 
                op(data_.get()[i], other.data_.get()[i]);
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
        return elementwiseOp(other, std::plus<T>(), "addition");
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
        return elementwiseOp(other, std::minus<T>(), "subtraction");
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
        return elementwiseOp(other, std::multiplies<T>(), "multiplication");
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
        // Custom lambda to check division by zero
        return elementwiseOp(other, [](const T& a, const T& b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        }, "division");
    }

    // inplace operations
    template <typename T>
    template <typename Op>
    Tensor<T>& Tensor<T>::inplaceElementwiseOp(
        const Tensor<T> &other, Op op, const char* opname) {
        // shallow copy
        auto a = *this; 
        auto b = other;

        if (shape_ != other.shape_) {
            // try to broadcast the shapes
            auto new_shape = compute_broadcast_shape(shape_, other.shape_);
            a = broadcast(a, new_shape);
            b = broadcast(b, new_shape);
        }

        if (a.shape_ != b.shape_) {
            throw std::invalid_argument(std::string("Shapes must match for ") + opname);
        }

        // a general version for all cases
        const int ndim = a.shape_.size();
        const int total = std::accumulate(
            a.shape_.begin(), a.shape_.end(), 1, 
            std::multiplies<>());

        // Compute strides for iteration
        const auto& a_shape = a.shape_;
        const auto& b_shape = b.shape_;
        const auto& a_strides = a.strides_;
        const auto& b_strides = b.strides_;

        auto* a_ptr = a.data_.get();
        auto* b_ptr = b.data_.get();

        // Iterate using flat index + unravel
        std::vector<size_t> indices(ndim, 0);

        for (size_t idx = 0; idx < total; ++idx) {
            // Compute flat index with broadcasting
            size_t a_offset = 0;
            size_t b_offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                a_offset += (a_shape[d] == 1 ? 0 : indices[d]) * a_strides[d];
                b_offset += (b_shape[d] == 1 ? 0 : indices[d]) * b_strides[d];
            }

            a_ptr[a_offset] = op(a_ptr[a_offset], b_ptr[b_offset]);

            // Advance multi-dimensional index
            for (ssize_t d = ndim - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < a_shape[d]) {
                    break;
                }
                indices[d] = 0;
            }
        }

        // shallow copy to this tensor (update the view)
        *this = a; 
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T> &other) {
        return inplaceElementwiseOp(
            other, std::plus<T>(), "addition");
    }
    
    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T> &other) {
        return inplaceElementwiseOp(
            other, std::minus<T>(), "subtraction");
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T> &other) {
        return inplaceElementwiseOp(
            other, std::multiplies<T>(), "multiplication");
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T> &other) {
        // Custom lambda to check division by zero
        return inplaceElementwiseOp(
            other, [](const T& a, const T& b) {
            if (b == 0) 
                throw std::runtime_error("Division by zero");
            return a / b;
        }, "division");
    }

} // namespace mtb
