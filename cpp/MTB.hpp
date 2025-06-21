#ifndef __MTB_HPP__
#define __MTB_HPP__

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
template <typename T>
class Tensor {
public:
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
        data_(other.data_) {}

    // Move constructor
    Tensor(Tensor &&other) noexcept
        :shape_(std::move(other.shape_)), 
        strides_(std::move(other.strides_)), 
        data_(std::move(other.data_)) {
        other.data_ = nullptr; 
        other.size_ = 0;
        other.strides_.clear();
        other.shape_.clear();
        // Nullify the moved-from object's data pointer
    }
    // Copy assignment operator, shallow copy
    Tensor& operator=(const Tensor &other) {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            data_ = other.data_;
        }
        return *this;
    }
    // Move assignment operator
    Tensor& operator=(Tensor &&other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            data_ = std::move(other.data_);
            other.data_ = nullptr; 
            other.size_ = 0;
            other.strides_.clear();
            other.shape_.clear();
            // Nullify the moved-from object's data pointer
        }
        return *this;
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
    // using (i) to access 1 dimension tensor
    T& operator()(int i) {
        if (i < 0 || i >= shape_[0]) {
            throw std::out_of_range("Index out of range");
        }
        return data_.get()[i];
    }
    // using (i, j) to access 2 dimension tensor
    T& operator()(int i, int j) {
        if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
            throw std::out_of_range("Index out of range");
        }
        return data_.get()[i * strides_[0] + j * strides_[1]];
    }
    // using (i, j, k) to access 3 dimension tensor
    T& operator()(int i, int j, int k) {
        if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1] || k < 0 || k >= shape_[2]) {
            throw std::out_of_range("Index out of range");
        }
        return data_.get()[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }
    // using (i, j, k, l) to access 4 dimension tensor
    T& operator()(int i, int j, int k, int l) {
        if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1] || k < 0 || k >= shape_[2] || l < 0 || l >= shape_[3]) {
            throw std::out_of_range("Index out of range");
        }
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

#endif // __MTB_HPP__