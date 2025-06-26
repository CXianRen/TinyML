#pragma once

#include <vector>
namespace mtb {

template <typename T>
class Tensor {
    public:
    // Constructor
    Tensor(const std::vector<int> &shape);

    Tensor(const std::vector<int> &shape, 
           const std::vector<T> &data);

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
    Tensor(const Tensor &other);

    // Move constructor
    Tensor(Tensor &&other) noexcept;

    // Copy assignment operator, shallow copy
    Tensor& operator=(const Tensor &other);

    // Move assignment operator
    Tensor& operator=(Tensor &&other) noexcept;

    // checker
    bool is_contiguous() const;

    // deep copy
    Tensor copy() const;

    // reshape
    Tensor& reshape(const std::vector<int> &new_shape);


    /* Op */
    // using [] to access a innner dimension tensor
    Tensor operator[](const int i) const;

    // using (i) to access 1 dimension tensor
    T& operator()(const int i) const;

    // using (i, j) to access 2 dimension tensor
    T& operator()(const int i, const int j) const;

    // using (i, j, k) to access 3 dimension tensor
    T& operator()(const int i, const int&  j,const int k) const;

    // using (i, j, k, l) to access 4 dimension tensor
    T& operator()(const int i, const int j, 
                  const int k, const int l) const;
    
    // math operations

    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;

    // inplace operations
    Tensor& operator+=(const Tensor &other);
    Tensor& operator-=(const Tensor &other);
    Tensor& operator*=(const Tensor &other);
    Tensor& operator/=(const Tensor &other);

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
    // setter
    void shape(const std::vector<int> &shape) {
        shape_ = shape;
    }

    void strides(const std::vector<int> &strides) {
        strides_ = strides;
    }

    void data(const std::shared_ptr<T[]> &data) {
        data_ = data;
    }

    void set_size(int size) {
        size_ = size;
    }

    private:
    template <typename Op>
    Tensor elementwiseOp(const Tensor &other, 
        Op op, const char* opname) const;

    template <typename Op>
    Tensor& inplaceElementwiseOp(
        const Tensor &other, Op op, const char* opname);

    // member 
    // shape of the tensor
    std::vector<int> shape_ = {}; 
    // strides for each dimension
    std::vector<int> strides_ = {}; 
    // smart pointer to data
    std::shared_ptr<T[]> data_ = nullptr;

    int size_ = 0;    
};

} // namespace mtb
