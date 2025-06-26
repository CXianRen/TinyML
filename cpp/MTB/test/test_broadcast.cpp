#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

void test_scalar_tensor() {
  START_TEST();
  TensorF t = mtb::scalar_to_tensor<float>(3.14f);
  assert(t.shape() == std::vector<size_t>({1}));
  assert(t.data()[0] == 3.14f);
  // print shape and strides
  std::cout << "Shape: " << t.shape() << ", Strides: "
            << t.strides() << std::endl;
  
  PASSLOG();
}

void test_broadcast() {
  START_TEST();
  // This function is not implemented yet
  // Placeholder for future implementation
  
  TensorF t1 = mtb::broadcast<float>(3.14f, {2, 3});
  assert(t1.shape() == std::vector<size_t>({2, 3}));
  assert(t1.strides() == std::vector<size_t>({0, 0}));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      assert(t1(i, j) == 3.14f);
    }
  }

  //  (m,1) -> (m,n)
  TensorF t2 = mtb::random<float>({3, 1});
  TensorF t3 = mtb::broadcast<float>(t2, {3, 4});
  assert(t3.shape() == std::vector<size_t>({3, 4}));
  assert(t3.strides() == std::vector<size_t>({1, 0}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      assert(t3(i, j) == t2(i, 0));
    }
  }

  // (1,n) -> (m,n)
  TensorF t4 = mtb::random<float>({1, 4});
  TensorF t5 = mtb::broadcast<float>(t4, {3, 4});
  assert(t5.shape() == std::vector<size_t>({3, 4}));
  assert(t5.strides() == std::vector<size_t>({0, 1}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      assert(t5(i, j) == t4(0, j));
    }
  }

  // (m,n) -> (m,n)
  TensorF t6 = mtb::random<float>({3, 4});
  TensorF t7 = mtb::broadcast<float>(t6, {3, 4});
  assert(t7.shape() == std::vector<size_t>({3, 4}));
  assert(t7.strides() == std::vector<size_t>({4, 1}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      assert(t7(i, j) == t6(i, j));
    }
  }

  // (1, c) -> (a, b, c)
  // (c, 1) -> (0, 0, 1)
  TensorF t8 = mtb::random<float>({1, 3});
  TensorF t9 = mtb::broadcast<float>(t8, {2, 3, 3});
  assert(t9.shape() == std::vector<size_t>({2, 3, 3}));
  assert(t9.strides() == std::vector<size_t>({0, 0, 1}));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        assert(t9(i, j, k) == t8(0, k));
      }
    }
  }

  // (m, n) -> (2, 2, m, n)
  TensorF t10 = mtb::random<float>({2, 3});
  TensorF t11 = mtb::broadcast<float>(t10, {2, 2, 2, 3});
  assert(t11.shape() == std::vector<size_t>({2, 2, 2, 3}));
  assert(t11.strides() == std::vector<size_t>({0, 0, 3, 1}));
  for (size_t i = 0; i < 2; ++i) 
    for (size_t j = 0; j < 2; ++j) 
      for (size_t k = 0; k < 2; ++k) 
        for (size_t l = 0; l < 3; ++l) 
          assert(t11(i, j, k, l) == t10(k, l));

  // 
  TensorF t12 = mtb::random<float>({3, 2});
  TensorF t13 = mtb::transpose(t12, {1, 0});
  TensorF t14 = mtb::broadcast<float>(t13, {2, 2, 2, 3});
  assert(t14.shape() == std::vector<size_t>({2, 2, 2, 3}));
  assert(t14.strides() == std::vector<size_t>({0, 0, 1, 2}));
  for (size_t i = 0; i < 2; ++i) 
    for (size_t j = 0; j < 2; ++j) 
      for (size_t k = 0; k < 2; ++k) 
        for (size_t l = 0; l < 3; ++l) 
          assert(t14(i, j, k, l) == t13(k, l));

  PASSLOG();
}

void test_compute_broadcast_shape(){
  START_TEST();
  
  float scalar = 3.14f;
  TensorF t1({2, 3});
  auto shape1 = mtb::compute_broadcast_shape(scalar, t1.shape());
  assert(shape1 == std::vector<size_t>({2, 3}));

  TensorF t2({3, 1});
  TensorF t3({3, 4});
  auto shape2 = mtb::compute_broadcast_shape(t2.shape(), t3.shape());
  assert(shape2 == std::vector<size_t>({3, 4}));

  TensorF t4({1, 4});
  TensorF t5({3, 4});
  auto shape3 = mtb::compute_broadcast_shape(t4.shape(), t5.shape());
  assert(shape3 == std::vector<size_t>({3, 4}));

  TensorF t6({3, 4, 5});
  TensorF t7({1,5});
  auto shape4 = mtb::compute_broadcast_shape(t6.shape(), t7.shape());
  assert(shape4 == std::vector<size_t>({3, 4, 5}));

  TensorF t8({2, 3});
  TensorF t9({2, 2, 2, 3});
  auto shape5 = mtb::compute_broadcast_shape(t8.shape(), t9.shape());
  assert(shape5 == std::vector<size_t>({2, 2, 2, 3}));

  PASSLOG();
}

int main(int argc, char** argv) {
  test_scalar_tensor();
  test_broadcast();
  test_compute_broadcast_shape();
  return 0;
}