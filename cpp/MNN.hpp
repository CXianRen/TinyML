#ifndef __MNN_HPP__
#define __MNN_HPP__

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace MNN {
    template<typename T>
    class Mat2D {
        public:
            Mat2D() : m_rows(0), m_cols(0), m_data(nullptr) {}

            Mat2D(int rows, int cols): m_rows(rows), m_cols(cols) {
                m_data = new T[rows * cols];
            }

            Mat2D& operator=(Mat2D&& other) noexcept {
                if (this != &other) {
                    delete[] m_data;
                    m_rows = other.m_rows;
                    m_cols = other.m_cols;
                    m_data = other.m_data;
                    other.m_data = nullptr;
                    other.m_rows = 0;
                    other.m_cols = 0;
                }
                return *this;
            }

            Mat2D& operator=(const Mat2D& other) {
                if (this != &other) {
                    delete[] m_data;
                    m_rows = other.m_rows;
                    m_cols = other.m_cols;
                    m_data = new T[m_rows * m_cols];
                    std::copy(other.m_data, other.m_data + m_rows * m_cols, m_data);
                }
                return *this;
            }

            Mat2D(Mat2D&& other) noexcept
                : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {
                other.m_data = nullptr;
                other.m_rows = 0;
                other.m_cols = 0;
            }

            ~Mat2D(){
                if (m_data)
                    delete[] m_data;
            }

            void set(int row, int col, T value) {
                if (row < 0 || row >= m_rows || col < 0 || col >= m_cols) {
                    throw std::out_of_range("Index out of bounds.");
                }
                m_data[row * m_cols + col] = value;
            }

            T get(int row, int col) const {
                if (row < 0 || row >= m_rows || col < 0 || col >= m_cols) {
                    throw std::out_of_range("Index out of bounds.");
                }
                return m_data[row * m_cols + col];
            }


            void multiply(const Mat2D<T>& other, Mat2D<T>& result) const {
                // [M,N] * [N,K] = [M,K]
                if (m_cols != other.m_rows || result.m_rows != m_rows || result.m_cols != other.m_cols) {
                    throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
                }
                int M = m_rows;
                int N = m_cols;
                int K = other.m_cols;

                for (int m = 0; m < M; m++) {
                    for (int k = 0; k < K; k++) {
                        for (int n = 0; n < N; n++) {
                            result.m_data[m*K + k] += 
                                m_data[m*N + n] * other.m_data[n*K + k];
                        }
                    }
                }
            }

            void add(const Mat2D<T>& other, Mat2D<T>& result) const {
                // [M,N] + [M,N] = [M,N]
                if (m_rows != other.m_rows || m_cols != other.m_cols || 
                    result.m_rows != m_rows || result.m_cols != m_cols) {
                    throw std::invalid_argument("Matrix dimensions do not match for addition.");
                }

                for (int i = 0; i < m_rows; ++i) {
                    for (int j = 0; j < m_cols; ++j) {
                        result.m_data[i * m_cols + j] = 
                            m_data[i * m_cols + j] + other.m_data[i * m_cols + j];
                    }
                }
            }

            Mat2D<T>& operator+=(const Mat2D<T>& other) {
                if (m_rows != other.m_rows || m_cols != other.m_cols) {
                    throw std::invalid_argument("Matrix dimensions do not match for addition.");
                }

                for (int i = 0; i < m_rows; ++i) {
                    for (int j = 0; j < m_cols; ++j) {
                        m_data[i * m_cols + j] += other.m_data[i * m_cols + j];
                    }
                }
                return *this;
            }

        // private:
            int m_rows;
            int m_cols;
            T* m_data; // Pointer to the data array
    };

    template<typename T>
    class MLinear {
        public:
            MLinear(int in_features, int out_features, bool bias = true){
                in_features_ = in_features;
                out_features_ = out_features;
                use_bias_ = bias;
                weight_ = Mat2D<T>(in_features, out_features);
                if (use_bias_) {
                    bias_ = Mat2D<T>(1, out_features);
                } else {
                    bias_ = Mat2D<T>(1, 0); // No bias
                }
            }
            ~MLinear() = default;

            void forward(Mat2D<T>& input, Mat2D<T>& output){
                input.multiply(weight_, output);
                if (use_bias_) {
                    output += bias_;
                }
                return;
            }

        // private:
            int in_features_;
            int out_features_;
            bool use_bias_;

            Mat2D<T> weight_; // Weight matrix
            Mat2D<T> bias_; // Bias vector, if used
    };

    template<typename T>
    class MEmbed {
        public:
            MEmbed(int vocab_size, int embed_dim) 
                : vocab_size_(vocab_size), embed_dim_(embed_dim) {
                embeddings_ = Mat2D<T>(vocab_size, embed_dim);
            }

            ~MEmbed() = default;

            void forward(std::vector<int> idxs, Mat2D<T>& output) {
                for (size_t i = 0; i < idxs.size(); ++i) {
                    if (idxs[i] < 0 || idxs[i] >= vocab_size_) {
                        throw std::out_of_range("Index out of bounds for embedding.");
                    }
                    for (int j = 0; j < embed_dim_; ++j) {
                        int idx = idxs[i];
                        output.set(i, j, embeddings_.get(idx, j));
                    }
                }
            }

        // private:
            int vocab_size_;
            int embed_dim_;
            Mat2D<T> embeddings_; // Embedding matrix
    };

    template<typename T>
    void softmax(Mat2D<T>& input, Mat2D<T>& output) {
        // only works for 2D matrix, and 
        // just for the last dimension
 
        for (int i = 0; i < input.m_rows; i++){
            // Find max for numerical stability      
            // out= exp(x - max(x))
            // out /= sum(out)

            T max_val = input.get(i, 0);
            for (int j = 1; j < input.m_cols; j++) {
                if (input.get(i, j) > max_val) {
                    max_val = input.get(i, j);
                }
            }
            
            for (int j = 0; j < input.m_cols; j++) {
                output.set(i, j, std::exp(input.get(i, j) - max_val));
            }

            T sum = 0;
            for (int j = 0; j < input.m_cols; j++) {
                sum += output.get(i, j);
            }
            // Normalize
            for (int j = 0; j < input.m_cols; j++) {
                output.set(i, j, output.get(i, j) / sum);
            }
        }
    }
}


#endif // __MNN_HPP__