#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

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


int main(int argc, char** argv) {
  test_zeros();
  test_ones();
  test_random();
  test_triu();
  return 0;
}