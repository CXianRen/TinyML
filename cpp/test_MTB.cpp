#include "MTB.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

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

// compare two vectors for equality
template <typename T>
bool compare_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (auto i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// test constructor
void test_constructor() {
  std::cout << "Testing constructor..." << std::endl;
  TensorF t1({10});        // 1D tensor
  TensorF t2({2, 3});      // 2D tensor
  TensorF t3({2, 3, 4});   // 3D tensor
  TensorF t4({2, 3, 4, 5});// 4D tensor
  
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t2 shape: " << t2.shape() << std::endl;
  std::cout << "t3 shape: " << t3.shape() << std::endl;
  std::cout << "t4 shape: " << t4.shape() << std::endl;

  // check the shapes
  if (!compare_vectors(t1.shape(), std::vector<int>{10})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape!");
  }
  if (!compare_vectors(t2.shape(), std::vector<int>{2, 3})) {
    throw std::runtime_error("Error: t2 shape does not match expected shape!");
  }
  if (!compare_vectors(t3.shape(), std::vector<int>{2, 3, 4})) {
    throw std::runtime_error("Error: t3 shape does not match expected shape!");
  }
  if (!compare_vectors(t4.shape(), std::vector<int>{2, 3, 4, 5})) {
    throw std::runtime_error("Error: t4 shape does not match expected shape!");
  }

  // check the sizes
  if (t1.size() != 10) {
    throw std::runtime_error("Error: t1 size does not match expected size!");
  }
  if (t2.size() != 6) {
    throw std::runtime_error("Error: t2 size does not match expected size!");
  }
  if (t3.size() != 24) {
    throw std::runtime_error("Error: t3 size does not match expected size!");
  }
  if (t4.size() != 120) {
     throw std::runtime_error("Error: t4 size does not match expected size!");
  }

  // check the data pointers
  if (t1.data().get() == nullptr) {
    throw std::runtime_error("Error: t1 data pointer is null!");
  }
  if (t2.data().get() == nullptr) {
    throw std::runtime_error("Error: t2 data pointer is null!");
  }
  if (t3.data().get() == nullptr) {
    throw std::runtime_error("Error: t3 data pointer is null!");
  }
  if (t4.data().get() == nullptr) {
    throw std::runtime_error("Error: t4 data pointer is null!");
  }

  // check the strides
  if (!compare_vectors(t1.strides(), std::vector<int>{1})) {
    throw std::runtime_error("Error: t1 strides do not match expected strides!");
  }
  if (!compare_vectors(t2.strides(), std::vector<int>{3, 1})) {
    throw std::runtime_error("Error: t2 strides do not match expected strides!");
    }
  if (!compare_vectors(t3.strides(), std::vector<int>{12, 4, 1})) {
    throw std::runtime_error("Error: t3 strides do not match expected strides!");
  }
  if (!compare_vectors(t4.strides(), std::vector<int>{60, 20, 5, 1})) {
    throw std::runtime_error("Error: t4 strides do not match expected strides!");
  }

  // copy constructor 
  TensorF t5(t1);
  std::cout << "t5 (copy of t1) shape: " << t5.shape() << std::endl;
  if (!compare_vectors(t1.shape(), t5.shape())) {
    throw std::runtime_error("Error: t5 shape does not match t1 shape!");
  }
  // compare contents // shallow copy
  if (t1.data().get() != t5.data().get()) {
    throw std::runtime_error("Error: t5 data does not match t1 data!");
  }

  // copy assignment operator
  TensorF t6 = t2;
  std::cout << "t6 (copy of t2) shape: " << t6.shape() << std::endl;
  if (!compare_vectors(t2.shape(), t6.shape())) {
    throw std::runtime_error("Error: t6 shape does not match t2 shape!");
  }
  // compare contents // shallow copy
  if (t2.data().get() != t6.data().get()) {
    throw std::runtime_error("Error: t6 data does not match t2 data!");
  }

  // move constructor
  TensorF t7(std::move(t3));
  std::cout << "t7 (moved from t3) shape: " << t7.shape() << std::endl;
  // t3 should be empty now
  // size should be 0
  if (t3.shape().size() != 0) {
    throw std::runtime_error("Error: t3 should be empty after move!");
  }
  // data should be nullptr
  if (t3.data().get() != nullptr) {
    throw std::runtime_error("Error: t3 data should be nullptr after move!");
 }
  // strides should be empty
  if (!t3.strides().empty()) {
    throw std::runtime_error("Error: t3 strides should be empty after move!");
  }

  // move assignment operator
  TensorF t8 = std::move(t4);
  std::cout << "t8 (moved from t4) shape: " << t8.shape() << std::endl;
  // t4 should be empty now
  if (t4.shape().size() != 0) {
    throw std::runtime_error("Error: t4 should be empty after move!");
  }
  // data should be nullptr
  if (t4.data().get() != nullptr) {
    throw std::runtime_error("Error: t4 data should be nullptr after move!");
  }
  // strides should be empty
  if (!t4.strides().empty()) {
    throw std::runtime_error("Error: t4 strides should be empty after move!");
  }

  std::cout << "All constructor tests passed!" << std::endl;
}

// test deep copy
void test_deep_copy() {
  std::cout << "Testing deep copy..." << std::endl;
  TensorF t1({2, 3});
  TensorF t2 = t1.copy(); // deep copy
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t2 shape: " << t2.shape() << std::endl;
  if (!compare_vectors(t1.shape(), t2.shape())) {
    // std::cerr << "Error: t2 shape does not match t1 shape!" << std::endl;
    throw std::runtime_error("Error: t2 shape does not match t1 shape!");
  }
  // compare contents // deep copy
  if (t1.data().get() == t2.data().get()) {
    throw std::runtime_error("Error: t2 data should not match t1 data after deep copy!");
  } 
  // compare strides
  if (!compare_vectors(t1.strides(), t2.strides())) {
    throw std::runtime_error("Error: t2 strides do not match t1 strides!");
 } 

  // compare data
  for (auto i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] != t2.data().get()[i]) {
        throw std::runtime_error("Error: t2 data does not match t1 data at index " + std::to_string(i) + "!");
    }
  }
  std::cout << "All deep copy tests passed!" << std::endl;
}

// test reshape
void test_reshape() {
  std::cout << "Testing reshape..." << std::endl;
  TensorF t1({2, 3});
  TensorF t2 = t1.reshape({3, 2}); // reshape
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t2 shape: " << t2.shape() << std::endl;
  if (!compare_vectors(t2.shape(), std::vector<int>{3, 2})) {
    throw std::runtime_error("Error: t2 shape does not match expected shape after reshape!");
  }
  // check strides
  if (!compare_vectors(t2.strides(), std::vector<int>{2, 1})) {
    throw std::runtime_error("Error: t2 strides do not match expected strides after reshape!");
  }
  std::cout << "All reshape tests passed!" << std::endl;
}

// test () operator
void test_operator() {
  std::cout << "Testing () operator..." << std::endl;
  
  // 1 D tensor
  TensorF t1({5});
  for (int i = 0; i < t1.size(); ++i) {
    t1(i) = static_cast<float>(i); // fill with values
  }
  // check values
  for (int i = 0; i < t1.size(); ++i) {
    if (t1(i) != static_cast<float>(i)) {
      throw std::runtime_error("Error: t1(" + std::to_string(i) + ") does not match expected value!");
    }
  }
  // 2 D tensor
  TensorF t2({2, 3});
  for (int i = 0; i < t2.shape()[0]; ++i) {
    for (int j = 0; j < t2.shape()[1]; ++j) {
      t2(i, j) = static_cast<float>(i * t2.shape()[1] + j); // fill with values
    }
  }
  // check values
  for (int i = 0; i < t2.shape()[0]; ++i) {
    for (int j = 0; j < t2.shape()[1]; ++j) {
      if (t2(i, j) != static_cast<float>(i * t2.shape()[1] + j)) {
        throw std::runtime_error("Error: t2(" + std::to_string(i) + ", " + std::to_string(j) + ") does not match expected value!");
      }
    }
  }
  // 3 D tensor
  TensorF t3({2, 2, 2});
  for (int i = 0; i < t3.shape()[0]; ++i) {
    for (int j = 0; j < t3.shape()[1]; ++j) {
      for (int k = 0; k < t3.shape()[2]; ++k) {
        t3(i, j, k) = static_cast<float>(i * t3.shape()[1] * t3.shape()[2] + j * t3.shape()[2] + k); // fill with values
      }
    }
  }
  // check values
  for (int i = 0; i < t3.shape()[0]; ++i) {
    for (int j = 0; j < t3.shape()[1]; ++j) {
      for (int k = 0; k < t3.shape()[2]; ++k) {
        if (t3(i, j, k) != static_cast<float>(i * t3.shape()[1] * t3.shape()[2] + j * t3.shape()[2] + k)) {
          throw std::runtime_error("Error: t3(" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ") does not match expected value!");
        }
      }
    }
  }
  // 4 D tensor
  TensorF t4({2, 2, 2, 2});
  for (int i = 0; i < t4.shape()[0]; ++i) {
    for (int j = 0; j < t4.shape()[1]; ++j) {
      for (int k = 0; k < t4.shape()[2]; ++k) {
        for (int l = 0; l < t4.shape()[3]; ++l) {
          t4(i, j, k, l) = static_cast<float>(i * t4.shape()[1] * t4.shape()[2] * t4.shape()[3] + j * t4.shape()[2] * t4.shape()[3] + k * t4.shape()[3] + l); // fill with values
        }
      }
    }
  }
  // check values
  for (int i = 0; i < t4.shape()[0]; ++i) {
    for (int j = 0; j < t4.shape()[1]; ++j) {
      for (int k = 0; k < t4.shape()[2]; ++k) {
        for (int l = 0; l < t4.shape()[3]; ++l) {
          if (t4(i, j, k, l) != static_cast<float>(i * t4.shape()[1] * t4.shape()[2] * t4.shape()[3] + j * t4.shape()[2] * t4.shape()[3] + k * t4.shape()[3] + l)) {
            throw std::runtime_error("Error: t4(" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + std::to_string(l) + ") does not match expected value!");
          }
        }
      }
    }
  }
  std::cout << "All () operator tests passed!" << std::endl;
} 

// Template test for binary operators (+, -, *, /)
template <typename Op, typename OpFunc>
void test_op_template(const std::string& op_name, OpFunc op_func) {
  std::cout << "Testing operator " << op_name << " ..." << std::endl;

  auto fill_and_check = [&](const std::vector<int>& shape) {
    TensorF t1(shape);
    TensorF t2(shape);
    // Fill tensors with values
    for (int i = 0; i < t1.size(); ++i) {
      t1.data().get()[i] = static_cast<float>(i);
      t2.data().get()[i] = static_cast<float>(i + 1);
    }
    TensorF t_result = op_func(t1, t2);
    if (!compare_vectors(t_result.shape(), shape)) {
      throw std::runtime_error(
        "Error: t_result shape does not match expected shape after " 
        + op_name + "!");
    }
    for (int i = 0; i < t_result.size(); ++i) {
      float expected = Op{}(t1.data().get()[i], t2.data().get()[i]);
      if (t_result.data().get()[i] != expected) {
        throw std::runtime_error(
          "Error: t_result(" + std::to_string(i) +
          ") does not match expected value for " + op_name + "!");
      }
    }
  };

  fill_and_check({5});         // 1D
  fill_and_check({2, 3});      // 2D
  fill_and_check({2, 3, 4});   // 3D
  fill_and_check({2, 3, 4, 5});// 4D
  std::cout << "All " << op_name << " operator tests passed!" << std::endl;
}

void test_op_add() {
  test_op_template<std::plus<float>>(
    "+", 
    [](const TensorF& a, const TensorF& b) { return a + b; }
  );
}
void test_op_sub() {
  test_op_template<std::minus<float>>("-", 
    [](const TensorF& a, const TensorF& b) { return a - b; }
  );
}
void test_op_mul() {
  test_op_template<std::multiplies<float>>("*", 
    [](const TensorF& a, const TensorF& b) { return a * b; });
}
void test_op_div() {
  test_op_template<std::divides<float>>("/", 
    [](const TensorF& a, const TensorF& b) { return a / b; });
}

// inplace operators
template <typename Op, typename OpFunc>
void test_inplace_op_template(const std::string& op_name, OpFunc op_func) {
  std::cout << "Testing inplace operator " << op_name << " ..." << std::endl;

  auto fill_and_check = [&](const std::vector<int>& shape) {
    TensorF t1(shape);
    TensorF t2(shape); // for comparison
    // Fill tensor with values
    for (int i = 0; i < t1.size(); ++i) {
      t1.data().get()[i] = static_cast<float>(i);
      t2.data().get()[i] = static_cast<float>(i + 1); 
      // t2 is added to t1 in the inplace operation
    }
    TensorF t1_copy = t1.copy(); // make a copy for comparison

    // Perform inplace operation
    op_func(t1, t2);
    if (!compare_vectors(t1.shape(), shape)) {
      throw std::runtime_error(
        "Error: t1 shape does not match expected shape after inplace " 
        + op_name + "!");
    }
    for (int i = 0; i < t1.size(); ++i) {
      float expected = Op{}(t1_copy.data().get()[i], t2.data().get()[i]);
      if (t1.data().get()[i] != expected) {
        throw std::runtime_error(
          "Error: t1(" + std::to_string(i) +
          ") does not match expected value for inplace " + op_name + "!");
      }
    }
  };

  fill_and_check({5});         // 1D
  fill_and_check({2, 3});      // 2D
  fill_and_check({2, 3, 4});   // 3D
  fill_and_check({2, 3, 4, 5});// 4D
  std::cout << "All inplace " << op_name << " operator tests passed!" << std::endl;
}

void test_inplace_op_add() {
  test_inplace_op_template<std::plus<float>>(
    "+=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 += t2; 
    }
  );
}
void test_inplace_op_sub() {
  test_inplace_op_template<std::minus<float>>("-=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 -= t2; 
    }
  );
}
void test_inplace_op_mul() {
  test_inplace_op_template<std::multiplies<float>>("*=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 *= t2; 
    }
  );
}
void test_inplace_op_div() {
  test_inplace_op_template<std::divides<float>>("/=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 /= t2; 
    }
  );
}

// test zeros 
void test_zeros() {
  std::cout << "Testing zeros..." << std::endl;
  TensorF t1 = mtb::zeros<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<int>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after zeros!");
  }
  for (int i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] != 0.0f) {
      throw std::runtime_error("Error: t1 data is not all zeros!");
    }
  }
  std::cout << "All zeros tests passed!" << std::endl;
}

// test ones
void test_ones() {
  std::cout << "Testing ones..." << std::endl;
  TensorF t1 = mtb::ones<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<int>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after ones!");
  }
  for (int i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] != 1.0f) {
      throw std::runtime_error("Error: t1 data is not all ones!");
    }
  }
  std::cout << "All ones tests passed!" << std::endl;
}

// test random
void test_random() {
  std::cout << "Testing random..." << std::endl;
  TensorF t1 = mtb::random<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<int>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after random!");
  }
  for (int i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] < 0.0f || t1.data().get()[i] > 1.0f) {
      throw std::runtime_error("Error: t1 data is not in the range [0, 1]!");
    }
  }
  std::cout << "All random tests passed!" << std::endl;
}

// test triu
void test_triu() {
  std::cout << "Testing triu..." << std::endl;
  TensorF t1({4, 4});
  for (int i = 0; i < t1.shape()[0]; ++i) {
    for (int j = 0; j < t1.shape()[1]; ++j) {
      t1(i, j) = static_cast<float>(i * t1.shape()[1] + j);
    }
  }
  TensorF t2 = mtb::triu(t1);
  if (!compare_vectors(t2.shape(), t1.shape())) {
    throw std::runtime_error("Error: t2 shape does not match t1 shape after triu!");
  }
  for (int i = 0; i < t2.shape()[0]; ++i) {
    for (int j = 0; j < t2.shape()[1]; ++j) {
      if (i > j && t2(i, j) != 0.0f) {
        throw std::runtime_error("Error: t2(" + 
          std::to_string(i) + 
          ", " + std::to_string(j) + 
          ") is not zero after triu!");
      }
      if (i <= j && t2(i, j) != t1(i, j)) {
        throw std::runtime_error(
          "Error: t2(" + std::to_string(i) + 
          ", " + std::to_string(j) + 
          ") does not match t1 value after triu!");
      }
    }
  }
  std::cout << "All triu tests passed!" << std::endl;
}

// test transpose
void test_transpose() {
  std::cout << "Testing transpose..." << std::endl;
  TensorF t1({2, 3});
  for (int i = 0; i < t1.shape()[0]; ++i) {
    for (int j = 0; j < t1.shape()[1]; ++j) {
      t1(i, j) = static_cast<float>(i * t1.shape()[1] + j);
    }
  }
  TensorF t2 = mtb::transpose(t1, {1, 0}); // transpose
  if (!compare_vectors(t2.shape(), std::vector<int>{3, 2})) {
    throw std::runtime_error(
      "Error: t2 shape does not match expected shape after transpose!");
  }

  // check strides
  if (!compare_vectors(t2.strides(), std::vector<int>{1, 3})) {
    throw std::runtime_error(
      "Error: t2 strides do not match expected strides after transpose!");
  }

  for (int i = 0; i < t2.shape()[0]; ++i) {
    for (int j = 0; j < t2.shape()[1]; ++j) {
      if (t2(i, j) != t1(j, i)) {
        throw std::runtime_error("Error: t2(" + 
          std::to_string(i) + 
          ", " + std::to_string(j) + 
          ") does not match t1 transposed value!");
      }
    }
  }
}

// test concatenation
void test_concat() {
  std::cout << "Testing concatenation..." << std::endl;
  TensorF t1({2, 3});
  TensorF t2({2, 3});
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t1 strides: " << t1.strides() << std::endl;
  std::cout << "size of t1: " << t1.size() << std::endl;

  // Fill tensors with values
  for (int i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i);
    t2.data().get()[i] = static_cast<float>(i + 6); // offset by 6
  }
  
  TensorF t_concat = mtb::concatenate<float>({t1, t2}, 0); // concatenate along first axis
  if (!compare_vectors(t_concat.shape(), std::vector<int>{4, 3})) {
    throw std::runtime_error("Error: t_concat shape does not match expected shape after concat!");
  }
  
  for (int i = 0; i < t_concat.shape()[0]; ++i) {
    for (int j = 0; j < t_concat.shape()[1]; ++j) {
      float expected_value = (i < 2) ? t1(i, j) : t2(i - 2, j);
      if (t_concat(i, j) != expected_value) {
        throw std::runtime_error("Error: t_concat(" + 
          std::to_string(i) + 
          ", " + std::to_string(j) + 
          ") does not match expected value after concat!");
      }
    }
  }
  
  // Check the strides
  if (!compare_vectors(t_concat.strides(), std::vector<int>{3, 1})) {
    throw std::runtime_error("Error: t_concat strides do not match expected strides after concat!");
  }

  std::cout << "All concatenation tests passed!" << std::endl;
}


int main(int argc, char** argv) {
  test_constructor();
  test_deep_copy();
  test_reshape();
  test_operator();
  test_op_add();
  test_op_sub();
  test_op_mul();
  test_op_div();
  test_inplace_op_add();
  test_inplace_op_sub();
  test_inplace_op_mul();
  test_inplace_op_div();
  // 
  test_zeros();
  test_ones();
  test_random();
  test_triu();
  test_transpose();
  test_concat();
  return 0;
}