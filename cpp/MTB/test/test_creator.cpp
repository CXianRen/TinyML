#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

// test zeros 
void test_zeros() {
  START_TEST();
  TensorF t1 = mtb::zeros<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<size_t>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after zeros!");
  }
  for (size_t i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] != 0.0f) {
      throw std::runtime_error("Error: t1 data is not all zeros!");
    }
  }
  PASSLOG();
}

// test ones
void test_ones() {
  START_TEST();
  TensorF t1 = mtb::ones<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<size_t>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after ones!");
  }
  for (size_t i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] != 1.0f) {
      throw std::runtime_error("Error: t1 data is not all ones!");
    }
  }
  PASSLOG();
}

// test random
void test_random() {
  START_TEST();
  TensorF t1 = mtb::random<float>({2, 3});
  if (!compare_vectors(t1.shape(), std::vector<size_t>{2, 3})) {
    throw std::runtime_error("Error: t1 shape does not match expected shape after random!");
  }
  for (size_t i = 0; i < t1.size(); ++i) {
    if (t1.data().get()[i] < 0.0f || t1.data().get()[i] > 1.0f) {
      throw std::runtime_error("Error: t1 data is not in the range [0, 1]!");
    }
  }
  PASSLOG();
}

// test triu
void test_triu() {
  START_TEST();
  TensorF t1({3, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f,
                      7.0f, 8.0f, 9.0f});

  TensorF t2 = mtb::triu(t1, 0);

  assert((t2.shape() == std::vector<size_t>{3, 3}));

  std::vector<float> expected_data = {1.0f, 2.0f, 3.0f,
                                      0.0f, 5.0f, 6.0f,
                                      0.0f, 0.0f, 9.0f};
  for (size_t i = 0; i < t2.size(); ++i) {
    assert(t2.data().get()[i] == expected_data[i]);
  }

  TensorF t3 = mtb::triu(t1, 1);
  assert((t3.shape() == std::vector<size_t>{3, 3}));
  std::vector<float> expected_data2 = {0.0f, 2.0f, 3.0f,
                                       0.0f, 0.0f, 6.0f,
                                       0.0f, 0.0f, 0.0f};
  for (size_t i = 0; i < t3.size(); ++i) {
    assert(t3.data().get()[i] == expected_data2[i]);
  }

  Tensor<uint8_t> t4({3, 3}, {1, 1 , 1,
                              1, 1, 1,
                              1, 1, 1});
  Tensor<uint8_t> t5 = mtb::triu(t4, 0);
  assert((t5.shape() == std::vector<size_t>{3, 3}));
  std::vector<uint8_t> expected_data3 = {1, 1, 1,
                                         0, 1, 1,
                                         0, 0, 1};
  
  for (size_t i = 0; i < t5.size(); ++i) {
    assert(t5.data().get()[i] == expected_data3[i]);
  }

  PASSLOG();
}


int main(int argc, char** argv) {
  test_zeros();
  test_ones();
  test_random();
  test_triu();
  return 0;
}