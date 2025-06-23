#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

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

void test_is_contiguous() {
  std::cout << "Testing is_contiguous..." << std::endl;
  TensorF t1({2, 3}); // contiguous
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t1 strides: " << t1.strides() << std::endl;
  if (!t1.is_contiguous()) {
    throw std::runtime_error("Error: t1 should be contiguous!");
  }
  TensorF t2=mtb::transpose(t1, {1, 0}); // non-contiguous
  std::cout << "t2 shape: " << t2.shape() << std::endl;
  std::cout << "t2 strides: " << t2.strides() << std::endl;
  if (t2.is_contiguous()) {
    throw std::runtime_error("Error: t2 should not be contiguous!");
  }
  std::cout << "[Passed] is_contiguous test!" << std::endl;
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


// test [] operator
void test_bracket_operator() {
  START_TEST();
  
  TensorF t1({2, 3});
  TensorF t2 = t1[0]; // get first row
  assert(t2.shape() == std::vector<int>{3});
  assert(t2.strides() == std::vector<int>{1});
  for (int i = 0; i < t2.size(); ++i) {
    assert(t2(i) == t1(0, i)); // check values
  }
  
  TensorF t3({2,2,3}, {
    1, 2, 3,
    4, 5, 6,

    7, 8, 9,
    10, 11, 12
  });

  TensorF t4 = t3[1]; // get second slice
  assert((t4.shape() == std::vector<int>{2, 3}));
  assert((t4.strides() == std::vector<int>{3, 1}));
  assert(t4(0, 0) == 7);
  assert(t4(0, 1) == 8);
  assert(t4(0, 2) == 9);
  assert(t4(1, 0) == 10);
  assert(t4(1, 1) == 11);
  assert(t4(1, 2) == 12);


  TensorF t5 = t4[1]; // get second row of t4
  assert(t5.shape() == std::vector<int>{3});
  assert(t5.strides() == std::vector<int>{1});
  assert(t5(0) == 10);
  assert(t5(1) == 11);
  assert(t5(2) == 12);

  PASSLOG();
}

int main(int argc, char** argv) {
  test_constructor();
  test_deep_copy();
  test_reshape();
  test_is_contiguous();
  test_operator();
  test_bracket_operator();
  return 0;
}