#include "test_common.hpp"

#include <iostream>
using namespace mtb;

typedef Tensor<float> TensorF;

// Template test for binary operators (+, -, *, /)
template <typename Op, typename OpFunc>
void test_op_template(const std::string& op_name, OpFunc op_func) {
  auto fill_and_check = [&](const std::vector<size_t>& shape) {
    TensorF t1(shape);
    TensorF t2(shape);
    // Fill tensors with values
    for (size_t i = 0; i < t1.size(); ++i) {
      t1.data().get()[i] = static_cast<float>(i);
      t2.data().get()[i] = static_cast<float>(i + 1);
    }
    TensorF t_result = op_func(t1, t2);
    if (!compare_vectors(t_result.shape(), shape)) {
      throw std::runtime_error(
        "Error: t_result shape does not match expected shape after " 
        + op_name + "!");
    }
    for (size_t i = 0; i < t_result.size(); ++i) {
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
}

void test_op_add() {
  START_TEST();

  test_op_template<std::plus<float>>(
    "+", 
    [](const TensorF& a, const TensorF& b) { return a + b; }
  );
  PASSLOG();
}
void test_op_sub() {
  START_TEST();
  test_op_template<std::minus<float>>("-", 
    [](const TensorF& a, const TensorF& b) { return a - b; }
  );
  PASSLOG();
}
void test_op_mul() {
  START_TEST();
  test_op_template<std::multiplies<float>>("*", 
    [](const TensorF& a, const TensorF& b) { return a * b; });
  PASSLOG();
}

void test_op_div() {
  START_TEST();
  test_op_template<std::divides<float>>("/", 
    [](const TensorF& a, const TensorF& b) { return a / b; });
  PASSLOG();
}

// inplace operators
template <typename Op, typename OpFunc>
void test_inplace_op_template(const std::string& op_name, OpFunc op_func) {
  std::cout << "Testing inplace operator " << op_name << " ..." << std::endl;

  auto fill_and_check = [&](const std::vector<size_t>& shape) {
    TensorF t1(shape);
    TensorF t2(shape); // for comparison
    // Fill tensor with values
    for (size_t i = 0; i < t1.size(); ++i) {
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
    for (size_t i = 0; i < t1.size(); ++i) {
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
}

void test_inplace_op_add() {
  START_TEST();
  test_inplace_op_template<std::plus<float>>(
    "+=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 += t2; 
    }
  );
  PASSLOG();
}

void test_inplace_op_sub() {
  START_TEST();
  test_inplace_op_template<std::minus<float>>("-=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 -= t2; 
    }
  );
  PASSLOG();
}

void test_inplace_op_mul() {
  START_TEST();
  test_inplace_op_template<std::multiplies<float>>("*=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 *= t2; 
    }
  );
  PASSLOG();
}

void test_inplace_op_div() {
  START_TEST();
  test_inplace_op_template<std::divides<float>>("/=", 
    [](TensorF& t1, const TensorF& t2) { 
      t1 /= t2; 
    }
  );
  PASSLOG();
}

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

// test concatenation
void test_concat() {
  START_TEST();
  TensorF t1({2, 3});
  TensorF t2({2, 3});
  std::cout << "t1 shape: " << t1.shape() << std::endl;
  std::cout << "t1 strides: " << t1.strides() << std::endl;
  std::cout << "size of t1: " << t1.size() << std::endl;

  // Fill tensors with values
  for (size_t i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i);
    t2.data().get()[i] = static_cast<float>(i + 6); // offset by 6
  }
  
  TensorF t_concat = mtb::concatenate<float>({t1, t2}, 0); // concatenate along first axis
  if (!compare_vectors(t_concat.shape(), std::vector<size_t>{4, 3})) {
    throw std::runtime_error("Error: t_concat shape does not match expected shape after concat!");
  }
  
  for (size_t i = 0; i < t_concat.shape()[0]; ++i) {
    for (size_t j = 0; j < t_concat.shape()[1]; ++j) {
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
  if (!compare_vectors(t_concat.strides(), std::vector<size_t>{3, 1})) {
    throw std::runtime_error("Error: t_concat strides do not match expected strides after concat!");
  }

  PASSLOG();
}

// test where
// TODO 

void test_max() {
  START_TEST();
  TensorF t1({2, 3});
  for (size_t i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i);
  }
  auto max_tensor = mtb::max(t1, 1); // max along axis 1
  if (!compare_vectors(max_tensor.shape(), std::vector<size_t>{2,1})) {
    throw std::runtime_error(
      "Error: max_tensor shape does not match expected shape after max!");
  }
  if (max_tensor(0, 0) != 2.0f || max_tensor(1, 0) != 5.0f) {
    throw std::runtime_error(
      "Error: max_tensor values do not match expected values after max!");
  }
  // 3D tensor
  TensorF t2({2, 3, 4});
  for (size_t i = 0; i < t2.size(); ++i) {
    t2.data().get()[i] = static_cast<float>(i);
  }
  auto max_tensor_3d = mtb::max(t2, 2); // max along axis 2
  if (!compare_vectors(max_tensor_3d.shape(), std::vector<size_t>{2, 3, 1})) {
    throw std::runtime_error(
      "Error: max_tensor_3d shape does not match expected shape after max!");
  }
  if (max_tensor_3d(0, 0, 0) != 3.0f || 
      max_tensor_3d(0, 1, 0) != 7.0f ||
      max_tensor_3d(0, 2, 0) != 11.0f ||
      max_tensor_3d(1, 0, 0) != 15.0f || 
      max_tensor_3d(1, 1, 0) != 19.0f ||
      max_tensor_3d(1, 2, 0) != 23.0f) {
    throw std::runtime_error(
      "Error: max_tensor_3d values do not match expected values after max!");
  }
  PASSLOG();
}

// test sum
void test_sum() {
  START_TEST();
  TensorF t1({2, 2, 3});
  for (size_t i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i + 1); // fill with values 1 to 12
  }
  TensorF t_sum = mtb::sum(t1, 2); // sum along axis 2
  if (!compare_vectors(t_sum.shape(), std::vector<size_t>{2, 2, 1})) {
    throw std::runtime_error("Error: t_sum shape does not match expected shape after sum!");
  }
  if (t_sum(0, 0, 0) != (1.0f + 2.0f + 3.0f) || 
      t_sum(0, 1, 0) != (4.0f + 5.0f + 6.0f) ||
      t_sum(1, 0, 0) != (7.0f + 8.0f + 9.0f) || 
      t_sum(1, 1, 0) != (10.0f + 11.0f + 12.0f)) {
    throw std::runtime_error("Error: t_sum values do not match expected values after sum!");
  }
  PASSLOG();
}

// test mean
void test_mean() {
  START_TEST();
  TensorF t1({2, 2, 3});
  for (size_t i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i + 1); // fill with values 1 to 12
  }
  TensorF t_mean = mtb::mean(t1, 2); // mean along axis 2
  if (!compare_vectors(t_mean.shape(), std::vector<size_t>{2, 2, 1})) {
    std::cerr << "t_mean shape: " << t_mean.shape() << std::endl;
    std::cerr << "Expected shape: {2, 2, 1}" << std::endl;
    throw std::runtime_error(
      "Error: t_mean shape does not match expected shape after mean!:");
  }
  if (t_mean(0, 0, 0) != (1.0f + 2.0f + 3.0f) / 3.0f || 
      t_mean(0, 1, 0) != (4.0f + 5.0f + 6.0f) / 3.0f ||
      t_mean(1, 0, 0) != (7.0f + 8.0f + 9.0f) / 3.0f || 
      t_mean(1, 1, 0) != (10.0f + 11.0f + 12.0f) / 3.0f) {
    throw std::runtime_error(
      "Error: t_mean values do not match expected values after mean!");
 }
 PASSLOG();
}

// test var
void test_var() {
  START_TEST();
  TensorF t1({2, 2, 3});
  for (size_t i = 0; i < t1.size(); ++i) {
    t1.data().get()[i] = static_cast<float>(i + 1); // fill with values 1 to 12
  }
  TensorF t_var = mtb::var(t1, 2); // var along axis 2
  if (!compare_vectors(t_var.shape(), std::vector<size_t>{2, 2, 1})) {
    throw std::runtime_error("Error: t_var shape does not match expected shape after var!");
  }

  auto compute_var = [](float i, float j, float k) {
    float mean = (i + j + k) / 3.0f;
    return ((i - mean) * (i - mean) + 
            (j - mean) * (j - mean) + 
            (k - mean) * (k - mean))/ 3;
  };

  if (t_var(0, 0, 0) != compute_var(1.0f, 2.0f, 3.0f) || 
      t_var(0, 1, 0) != compute_var(4.0f, 5.0f, 6.0f) ||
      t_var(1, 0, 0) != compute_var(7.0f, 8.0f, 9.0f) || 
      t_var(1, 1, 0) != compute_var(10.0f, 11.0f, 12.0f)) {
    throw std::runtime_error("Error: t_var values do not match expected values after var!");
  };

  PASSLOG();
}

template <typename Func, typename StdFunc>
void test_unary_op(const std::string& name, 
  Func tensor_func, 
  StdFunc std_func) {
  TensorF t({2, 3});
  for (size_t i = 0; i < t.size(); ++i) {
    t.data().get()[i] = static_cast<float>(i + 1); // covers negative and positive
  }
  TensorF t_result = tensor_func(t);
  if (!compare_vectors(t_result.shape(), t.shape())) {
    throw std::runtime_error("Error: t_result shape does not match t shape after " + name + "!");
  }
  for (size_t i = 0; i < t_result.size(); ++i) {
    float expected = std_func(t.data().get()[i]);
    if (t_result.data().get()[i] != expected) {
      throw std::runtime_error("Error: t_result(" + std::to_string(i) + ") does not match expected value after " + name + "!");
    }
  }
}

void test_exp() {
  START_TEST();
  test_unary_op("exp", 
    [](const TensorF& t){ return mtb::exp(t); }, 
    [](float x){ return std::exp(x); });
  PASSLOG();
}

void test_sqrt() {
  START_TEST();
  test_unary_op("sqrt", 
    [](const TensorF& t){ return mtb::sqrt(t); }, 
    [](float x){ return std::sqrt(x); });
  PASSLOG();
}

void test_tanh() {
  START_TEST();
  test_unary_op("tanh", 
    [](const TensorF& t){ return mtb::tanh(t); }, 
    [](float x){ return std::tanh(x); });
  PASSLOG();
}

void test_softmax(){
  START_TEST();
  TensorF t({3}, {1, 2, 3});
  TensorF t_softmax = mtb::softmax(t);
  assert(t_softmax.shape() == t.shape());
  float expected[] = {0.09003057f, 0.24472847f, 0.66524096f};
  for (size_t i = 0; i < t_softmax.size(); ++i) {
    if (std::abs(t_softmax.data().get()[i] - expected[i]) > 1e-6) {
      throw std::runtime_error("Error: t_softmax(" + std::to_string(i) + 
        ") does not match expected value after softmax!");
    }
  }

  // 2d
  TensorF t2({2, 3}, {1, 2, 3, 4, 5, 7});
  TensorF t_softmax2 = mtb::softmax(t2, 1);
  assert(t_softmax2.shape() == t2.shape());
  float expected2[] = {0.09003057f, 0.24472847f, 0.66524096f,
                       0.04201007f, 0.1141952f,  0.84379473f};
  for (size_t i = 0; i < t_softmax2.size(); ++i) {
    if (std::abs(t_softmax2.data().get()[i] - expected2[i]) > 1e-6) {
      throw std::runtime_error("Error: t_softmax2(" + std::to_string(i) + 
        ") does not match expected value after softmax!");
    }
  }
  
  TensorF t3= t2.reshape({1, 1, 2, 3});
  TensorF t_softmax3 = mtb::softmax(t3, -1);
  assert(t_softmax3.shape() == t3.shape());
  float expected3[] = {0.09003057f, 0.24472847f, 0.66524096f,
                       0.04201007f, 0.1141952f,  0.84379473f};
  for (size_t i = 0; i < t_softmax3.size(); ++i) {
    if (std::abs(t_softmax3.data().get()[i] - expected3[i]) > 1e-6) {
      throw std::runtime_error("Error: t_softmax3(" + std::to_string(i) + 
        ") does not match expected value after softmax!");
    }
  }

  PASSLOG();
}

int main(int argc, char** argv) {
  test_op_add();
  test_op_sub();
  test_op_mul();
  test_op_div();
  test_inplace_op_add();
  test_inplace_op_sub();
  test_inplace_op_mul();
  test_inplace_op_div();
  
  // 
  test_max();
  test_sum();
  test_mean();
  test_var();

  test_exp();
  test_sqrt();
  test_tanh();

  // 
  test_softmax();
  return 0;
}