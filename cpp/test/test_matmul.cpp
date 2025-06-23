#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

void test_GEMM_2d(){
  START_TEST();

  TensorF a({3, 2}, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
          });

  TensorF b({2, 3}, {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
          });
  TensorF c = mtb::zeros<float>({3, 3});
  mtb::_GEMM(a, b, c.data().get());

  float expected[] = {
    9.0f, 12.0f, 15.0f,
    19.0f, 26.0f, 33.0f,
    29.0f, 40.0f, 51.0f
  };
  
  for (int i = 0; c.size() > i; ++i) {
    if (c.data()[i] != expected[i]) {
      std::cerr << "GEMM test failed at index " << i << ": expected " 
                << expected[i] << ", got " << c.data()[i] << std::endl;
      return;
    }
  }

  PASSLOG();
}


void test_GEMM_2d_transpose(){
  START_TEST();

  TensorF a({3, 2}, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
          });
  TensorF b({3, 2}, {
            1.0f, 4.0f,
            2.0f, 5.0f,
            3.0f, 6.0f
          });
  TensorF c = mtb::transpose<float>(b, {1, 0});
  TensorF d = mtb::zeros<float>({3, 3});
  mtb::_GEMM(a, c, d.data().get());

  float expected[] = {
    9.0f, 12.0f, 15.0f,
    19.0f, 26.0f, 33.0f,
    29.0f, 40.0f, 51.0f
  };

  for (int i = 0; d.size() > i; ++i) {
    if (d.data()[i] != expected[i]) {
      std::cerr << "GEMM transpose test failed at index " << i << ": expected " 
                << expected[i] << ", got " << d.data()[i] << std::endl;
      return;
    }
  }
  PASSLOG();
}

int main(int argc, char** argv) {
  test_GEMM_2d();
  test_GEMM_2d_transpose();
  return 0;
}