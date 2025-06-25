#ifndef __MTB_HPP__
#define __MTB_HPP__

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (int i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}


namespace mtb {

template <typename T>
class Tensor;

template <typename T>
Tensor<T> transpose(Tensor<T>& tensor, std::vector<int> axes);

template <typename T>
Tensor<T> broadcast(const Tensor<T> &tensor, 
                 const std::vector<int> &shape);

            
template <typename T>
class Tensor {
    public:
    friend Tensor<T> transpose<>(Tensor<T> &tensor, 
                std::vector<int> axes);
    friend Tensor<T> broadcast<>(const Tensor<T> &tensor, 
                const std::vector<int> &shape);

    // Constructor
    Tensor(const std::vector<int> &shape)
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

    Tensor(const std::vector<int> &shape, 
                const std::vector<T> &data)
        :Tensor(shape) {
        if (data.size() != size_) {
            throw std::invalid_argument(
                "Data size must match the total size of the shape");
        }
        // Copy data into the tensor
        memcpy(data_.get(), data.data(), size_ * sizeof(T));
    }

    // Destructor
    ~Tensor() {}

    // Copy constructor, shallow copy
    /*
        Tensor a;
        Tensor b = a; // copy assignment
       
        Tensor c = std::move(a); // move assignment

        Tensor d;
        d = b; // copy assignment
        d = std::move(c); // move assignment

    */
    Tensor(const Tensor &other)
        :shape_(other.shape_), 
        strides_(other.strides_), 
        size_(other.size_),
        data_(other.data_) {}

    // Move constructor
    Tensor(Tensor &&other) noexcept
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
    Tensor& operator=(const Tensor &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        data_ = other.data_;
        size_ = other.size_;
        return *this;
    }
    // Move assignment operator
    Tensor& operator=(Tensor &&other) noexcept {
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
    
    bool is_contiguous() const {
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
    Tensor copy() const {
        Tensor c(shape_);
        c.data_ = std::shared_ptr<T[]>(
            new T[size_], 
            std::default_delete<T[]>());
        memcpy(c.data_.get(), data_.get(), size_ * sizeof(T));
        return c;
    }

    // reshape
    Tensor& reshape(const std::vector<int> &new_shape) {
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
    Tensor operator[](const int i) const {
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
    T& operator()(const int i) {
        // if (i < 0 || i >= shape_[0]) {
        //     throw std::out_of_range("Index out of range");
        // }
        return data_.get()[i * strides_[0]];
    }

    // using (i, j) to access 2 dimension tensor
    T& operator()(const int i, const int j) {
        // if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
        //     throw std::out_of_range("Index out of range");
        // }
        return data_.get()[i * strides_[0] + j * strides_[1]];
    }

    // using (i, j, k) to access 3 dimension tensor
    T& operator()(const int i, const int&  j,const int k) {
        // if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1] || k < 0 || k >= shape_[2]) {
        //     throw std::out_of_range("Index out of range");
        // }
        return data_.get()[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }
    // using (i, j, k, l) to access 4 dimension tensor
    T& operator()(const int i, const int j, 
                    const int k, const int l) {
        // if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1] || k < 0 || k >= shape_[2] || l < 0 || l >= shape_[3]) {
        //     throw std::out_of_range("Index out of range");
        // }
        return data_.get()[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }
    
    // math operations
    // Helper for elementwise operations
    template <typename Op>
    Tensor elementwiseOp(const Tensor &other, Op op, const char* opname) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(std::string("Shapes must match for ") + opname);
        }
        Tensor result(shape_);
        for (int i = 0; i < size_; ++i) {
            result.data_.get()[i] = op(data_.get()[i], other.data_.get()[i]);
        }
        return result;
    }

    Tensor operator+(const Tensor &other) const {
        return elementwiseOp(other, std::plus<T>(), "addition");
    }

    Tensor operator-(const Tensor &other) const {
        return elementwiseOp(other, std::minus<T>(), "subtraction");
    }

    Tensor operator*(const Tensor &other) const {
        return elementwiseOp(other, std::multiplies<T>(), "multiplication");
    }

    Tensor operator/(const Tensor &other) const {
        // Custom lambda to check division by zero
        return elementwiseOp(other, [](const T& a, const T& b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        }, "division");
    }

    // inplace operations
    template <typename Op>
    Tensor& inplaceElementwiseOp(const Tensor &other, Op op, const char* opname) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(std::string("Shapes must match for ") + opname);
        }
        for (int i = 0; i < size_; ++i) {
            data_.get()[i] = op(data_.get()[i], other.data_.get()[i]);
        }
        return *this;
    }

    Tensor& operator+=(const Tensor &other) {
        return inplaceElementwiseOp(other, std::plus<T>(), "addition");
    }
    Tensor& operator-=(const Tensor &other) {
        return inplaceElementwiseOp(other, std::minus<T>(), "subtraction");
    }
    Tensor& operator*=(const Tensor &other) {
        return inplaceElementwiseOp(other, std::multiplies<T>(), "multiplication");
    }
    Tensor& operator/=(const Tensor &other) {
        // Custom lambda to check division by zero
        return inplaceElementwiseOp(other, [](const T& a, const T& b) {
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        }, "division");
    }

    // getter
    const std::vector<int>& shape() const {
        return shape_;
    }
    const std::vector<int>& strides() const {
        return strides_;
    }
    std::shared_ptr<T[]> data() const {
        return data_;
    }
    int size() const {
        return size_;
    }

    private:
    // member 
    std::vector<int> shape_ = {}; // shape of the tensor
    std::vector<int> strides_ = {}; // strides for each dimension
    // smart pointer to data
    std::shared_ptr<T[]> data_ = nullptr;

    int size_ = 0;    
};


// auto broadcast function
template <typename T>
std::vector<int> compute_broadcast_shape(
    const T &scalar,
    const std::vector<int> &shape) {
    return shape; // scalar can be broadcasted to any shape
}

std::vector<int> compute_broadcast_shape(
    const std::vector<int> &shape1,
    const std::vector<int> &shape2) {

    auto max_size = std::max(shape1.size(), shape2.size());
    std::vector<int> result_shape(max_size, 1);
    for (size_t i = 0; i < max_size; ++i) {
        int dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

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
    std::vector<int> shape = {1};
    Tensor<T> tensor(shape);
    // Assign the value to the tensor
    tensor(0) = value;
    return tensor;
}

template <typename T>
Tensor<T> broadcast(const Tensor<T> &tensor, 
                 const std::vector<int> &shape) {
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
    std::vector<int> r_shape = result.shape();
    std::vector<int> r_strides = result.strides();
    
    int padding = shape.size() - r_shape.size();
    for (int i = 0; i < padding; ++i) {
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
    result.shape_ = r_shape;
    result.strides_ = r_strides;

    return result;
}

template <typename T>
Tensor<T> broadcast(const T &scalar, 
                    const std::vector<int> &shape){
    // Convert scalar to tensor
    Tensor<T> tensor = scalar_to_tensor(scalar);
    // Broadcast the tensor to the new shape
    return broadcast(tensor, shape);
}

// Create tensors
template <typename T>
Tensor<T> zeros(std::vector<int> shape) {
    Tensor<T> tensor(shape);
    // Initialize all elements to zero
    std::fill(tensor.data().get(),  
            tensor.data().get() + tensor.size(), 
            T(0));
    return tensor;
}

template <typename T>
Tensor<T> ones(std::vector<int> shape) {
    Tensor<T> tensor(shape);
    // Initialize all elements to one
    std::fill(tensor.data().get(), 
            tensor.data().get() + tensor.size(), 
            T(1));
    return tensor;
}

template <typename T>
Tensor<T> random(std::vector<int> shape, 
                 T min = T(0), T max = T(1)) {
    Tensor<T> tensor(shape);
    // Initialize with random values in the range [min, max)
    for (int i = 0; i < tensor.size(); ++i) {
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

    int M = tensor.shape()[0];
    int N = tensor.shape()[1];

    int s0 = tensor.strides()[0];
    int s1 = tensor.strides()[1];
    int r0 = result.strides()[0];
    int r1 = result.strides()[1];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j >= i + k) {
                d_ptr[i * r0 + j * r1] = s_ptr[i * s0 + j * s1];
            } else {
                d_ptr[i * r0 + j * r1] = T(0);
            }
        }
    }
    return result;
}


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
    new_tensor.shape_ = new_shape;
    new_tensor.strides_ = new_strides;
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
        if (tensor.size() != tensor.shape()[0] * tensor.shape()[1]) {
            throw std::invalid_argument(
                "Tensor size does not match its shape,"\
                " expected size: " + std::to_string(tensor.shape()[0] * tensor.shape()[1]) +
                ", actual size: " + std::to_string(tensor.size()));
        }
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

// matmul function 2D
template <typename T>
void _GEMM(const Tensor<T>& a, const Tensor<T>& b, T* data){
    // a: [M, N], b: [N, K] -> result: [M, K]
 
    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[1];

    int a_stride0 = a.strides()[0];
    int a_stride1 = a.strides()[1];
    int b_stride0 = b.strides()[0];
    int b_stride1 = b.strides()[1];

    auto a_ptr = a.data().get();
    auto b_ptr = b.data().get();

    // Initialize the result tensor
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < N; ++k) {
                data[i * K + j] += 
                    a_ptr[i * a_stride0 + k * a_stride1] * 
                    b_ptr[k * b_stride0 + j * b_stride1];
            }
        }
    }
}

std::vector<std::vector<int>> generate_batch_indices(
        const std::vector<int>& shape) {
    std::vector<std::vector<int>> result;
    
    int ndim = shape.size();
    
    std::vector<int> indices(ndim, 0);
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
    return std::move(result);
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
    for (int i = 0; i < a_shape.size(); i++){
        new_a_shape[max_shape.size() - 1 - i] = 
            a_shape[a_shape.size() - 1 - i];
    }
    for (int i = 0; i < b_shape.size(); i++){
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
    int M = a_shape[a_shape.size() - 2];
    int N = a_shape[a_shape.size() - 1];
    int K = b_shape[b_shape.size() - 1];

    std::vector<int> result_shape = a_shape;
    result_shape[result_shape.size() - 2] = M;
    result_shape[result_shape.size() - 1] = K;

    Tensor<T> result = mtb::zeros<T>(result_shape);

    // generate batch shape
    std::vector<int> batch_shape =
        std::vector<int>(a_shape.begin(), 
                         a_shape.end() - 2);
    
    // generate indexes for batch dimensions
    auto batch_indices = generate_batch_indices(batch_shape);

    // Always use the batch loop, even if batch_indices is empty (no batch dims)
    for (int i = 0; i < std::max(1, static_cast<int>(batch_indices.size())); ++i) {
        std::vector<int> indices;
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

} // namespace mtb

#endif // __MTB_HPP__