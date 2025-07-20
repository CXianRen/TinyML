#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

// test concatenation
void test_concat() {
  START_TEST();
  // concat along dim == 0
  {
    TensorF t1({2, 3}, {
      0.0f, 1.0f, 2.0f,
      3.0f, 4.0f, 5.0f
    });
    TensorF t2({2, 3}, {
      6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f
    });

    float expect[12] = {
      // Batch 0
      0.0f, 1.0f, 2.0f,   // row 0
      3.0f, 4.0f, 5.0f,   // row 1
      6.0f, 7.0f, 8.0f,   // row 2 (from t2)
      9.0f, 10.0f, 11.0f  // row 3 (from t2)
    };

    // concatenate along first axis
    TensorF t_concat = mtb::concatenate<float>({t1, t2}, 0); 
    if (!compare_vectors(t_concat.shape(), std::vector<size_t>{4, 3})) {
      throw std::runtime_error(
        "Error: t_concat shape does not match expected shape after concat!");
    }
    
    for (size_t i = 0; i < t_concat.size(); ++i) {
      if (t_concat.data().get()[i] != expect[i]) {
        throw std::runtime_error(
          "Error: t_concat(" + std::to_string(i) + 
          ") does not match expected value after concat!");
      }
    }
    
    // Check the strides
    if (!compare_vectors(t_concat.strides(), std::vector<size_t>{3, 1})) {
      throw std::runtime_error(
        "Error: t_concat strides do not match expected strides after concat!");
    }
  }
  // concat along dim == 1
  {
    TensorF t1({2, 2, 3}, {
      1.0f, 2.0f, 3.0f, 
      4.0f, 5.0f, 6.0f, 

      7.0f, 8.0f, 9.0f, 
      10.0f, 11.0f, 12.0f
    });
    TensorF t2({2, 2, 3}, {
      13.0f, 14.0f, 15.0f, 
      16.0f, 17.0f, 18.0f, 

      19.0f, 20.0f, 21.0f,
      22.0f, 23.0f, 24.0f
    });
    
    float expect[24] = {
      // Batch 0
      1.0f, 2.0f, 3.0f,   // row 0
      4.0f, 5.0f, 6.0f,   // row 1
      13.0f,14.0f,15.0f,  // row 2 (from t2)
      16.0f,17.0f,18.0f,  // row 3 (from t2)

      // Batch 1
      7.0f, 8.0f, 9.0f,   // row 0
      10.0f,11.0f,12.0f,  // row 1
      19.0f,20.0f,21.0f,  // row 2 (from t2)
      22.0f,23.0f,24.0f   // row 3 (from t2)
    };
    
    
    TensorF t_concat = mtb::concatenate<float>({t1, t2}, 1);
    if (!compare_vectors(t_concat.shape(), 
      std::vector<size_t>{2, 4, 3})) {
      throw std::runtime_error(
        "Error: t_concat shape does not match expected shape after concat!");
    }
    for (size_t i = 0; i < t_concat.size(); ++i) {
      if (t_concat.data().get()[i] != expect[i]) {
        throw std::runtime_error(
          "Error: t_concat(" + std::to_string(i) + 
          ") does not match expected value after concat!");
      }
    }
  }
  // concat along dim == 2
  {
    TensorF t1({1, 2, 2, 2}, {
      1.0f, 2.0f, 
      3.0f, 4.0f,

      5.0f, 6.0f, 
      7.0f, 8.0f
    });

    TensorF t2({1, 2, 1, 2}, {
      9.0f, 10.0f, 
      11.0f, 12.0f
    });

    TensorF t_concat = mtb::concatenate<float>({t1, t2}, 2);

    float expect[16] = {
      1.0, 2.0,
      3.0, 4.0,
      9.0, 10.0,

      5.0, 6.0,
      7.0, 8.0,
      11.0, 12.0
    };

    if (!compare_vectors(t_concat.shape(), std::vector<size_t>{1, 2, 3, 2})) {
      throw std::runtime_error(
        "Error: t_concat shape does not match expected shape after concat!");
    }
    for (size_t i = 0; i < t_concat.size(); ++i) {
      if (t_concat.data().get()[i] != expect[i]) {
        throw std::runtime_error(
          "Error: t_concat(" + std::to_string(i) + 
          ") does not match expected value after concat!");
      }
    }
  }
  PASSLOG();
}

int main(int argc, char** argv) {
  test_concat();
  return 0;
}