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
  
  for (size_t i = 0; c.size() > i; ++i) {
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

  for (size_t i = 0; d.size() > i; ++i) {
    if (d.data()[i] != expected[i]) {
      std::cerr << "GEMM transpose test failed at index " << i << ": expected " 
                << expected[i] << ", got " << d.data()[i] << std::endl;
      return;
    }
  }
  PASSLOG();
}

void test_generate_batch_indices() {
  START_TEST();
  // This function is not implemented yet
  // Placeholder for future implementation
  
  // Example usage of generate_batch_indices
  auto indices = mtb::generate_batch_indices({2,2});
  
  assert(indices.size() == 4);
  assert(indices[0] == std::vector<size_t>({0, 0}));
  assert(indices[1] == std::vector<size_t>({0, 1}));
  assert(indices[2] == std::vector<size_t>({1, 0}));
  assert(indices[3] == std::vector<size_t>({1, 1}));

  auto indices2 = mtb::generate_batch_indices({3, 2, 2});
  assert(indices2.size() == 12);
  assert(indices2[0] == std::vector<size_t>({0, 0, 0}));
  assert(indices2[1] == std::vector<size_t>({0, 0, 1}));
  assert(indices2[2] == std::vector<size_t>({0, 1, 0}));
  assert(indices2[3] == std::vector<size_t>({0, 1, 1}));
  assert(indices2[4] == std::vector<size_t>({1, 0, 0}));
  assert(indices2[5] == std::vector<size_t>({1, 0, 1}));
  assert(indices2[6] == std::vector<size_t>({1, 1, 0}));
  assert(indices2[7] == std::vector<size_t>({1, 1, 1}));
  assert(indices2[8] == std::vector<size_t>({2, 0, 0}));
  assert(indices2[9] == std::vector<size_t>({2, 0, 1}));
  assert(indices2[10] == std::vector<size_t>({2, 1, 0}));
  assert(indices2[11] == std::vector<size_t>({2, 1, 1}));

  PASSLOG();
}

// test matmul 
void test_matmul() {
  START_TEST();
  {
    TensorF a({2, 3}, 
          { 1.0f, 2.0f, 3.0f, 
            4.0f, 5.0f, 6.0f});

    TensorF b({3, 2}, 
          { 1.0f, 4.0f, 
            2.0f, 5.0f, 
            3.0f, 6.0f});
    
    
    TensorF c = mtb::matmul(a, b);
    
    float expected[] = {
      14.0f, 32.0f,
      32.0f, 77.0f
    };
    
    for(size_t i = 0; i < c.size(); ++i) {
      assert(c.data()[i] == expected[i]);
    }
  }
  {
    TensorF a({2, 2, 3}, 
          { 1.0f, 2.0f, 3.0f, 
            4.0f, 5.0f, 6.0f,

            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f});
    TensorF b({2, 3, 2},
          { 1.0f, 4.0f,
            2.0f, 5.0f,
            3.0f, 6.0f,
            
            7.0f, 10.0f,
            8.0f, 11.0f,
            9.0f, 12.0f});
    TensorF c = mtb::matmul(a, b);
    float expected[] = {
      14.0f, 32.0f,
      32.0f, 77.0f,

      194.0f, 266.0f,
      266.0f, 365.0f
      // (49.0f + 72.0f + 81.0f), (70.0f + 88.0f + 108.0f),
      // (70.0f + 88.0f + 108.0f), (100.0f + 121.0f + 144.0f)
    };
    assert(c.shape() == std::vector<size_t>({2, 2, 2}));
    assert(c.strides() == std::vector<size_t>({4, 2, 1}));
    assert(c.size() == 8);

    for(size_t i = 0; i < c.size(); ++i) {
      if (c.data()[i] != expected[i]) {
        std::cerr << "Matmul test failed at index " << i << ": expected " 
                  << expected[i] << ", got " << c.data()[i] << std::endl;
        throw std::runtime_error("Matmul test failed");
      }
    }
  }

  PASSLOG();
}

int main(int argc, char** argv) {
  test_GEMM_2d();
  test_GEMM_2d_transpose();
  test_generate_batch_indices();
  test_matmul();
  return 0;
}