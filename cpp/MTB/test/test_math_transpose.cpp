#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

// test transpose
void test_transpose() {
  START_TEST();
  TensorF t1({2, 3});
  for (size_t i = 0; i < t1.shape()[0]; ++i) {
    for (size_t j = 0; j < t1.shape()[1]; ++j) {
      t1(i, j) = static_cast<float>(i * t1.shape()[1] + j);
    }
  }
  TensorF t2 = mtb::transpose(t1, {1, 0}); // transpose
  if (!compare_vectors(t2.shape(), std::vector<size_t>{3, 2})) {
    throw std::runtime_error(
      "Error: t2 shape does not match expected shape after transpose!");
  }

  // check strides
  if (!compare_vectors(t2.strides(), std::vector<size_t>{1, 3})) {
    throw std::runtime_error(
      "Error: t2 strides do not match expected strides after transpose!");
  }

  for (size_t i = 0; i < t2.shape()[0]; ++i) {
    for (size_t j = 0; j < t2.shape()[1]; ++j) {
      if (t2(i, j) != t1(j, i)) {
        throw std::runtime_error("Error: t2(" + 
          std::to_string(i) + 
          ", " + std::to_string(j) + 
          ") does not match t1 transposed value!");
      }
    }
  }
  PASSLOG();
}

int main(int argc, char** argv) { 
  test_transpose();
  return 0;
}