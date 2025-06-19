
#include <iostream>
#define private public // For testing purposes, to access private members in tests
#include "MNN.hpp"
#undef private // Restore private access

using namespace MNN;
using namespace std;

#define epsilon 1e-6

void testMat2D(){
    cout << "Mat1:" << std::endl;
    Mat2D<float> mat(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            mat.set(i, j, i * 2 + j);
            cout << mat.get(i, j) << " ";
        }
        cout << std::endl;
    }

    // 0 1      0 1 2
    // 2 3  *   3 4 5  

    // 3  4  5
    // 9 14 19

    cout << "Mat2:" << std::endl;
    Mat2D<float> mat2(2,3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat2.set(i, j, i * 3 + j);
            cout << mat2.get(i, j) << " ";
        }
        cout << std::endl;
    }

    Mat2D<float> mat3(2,3);
    mat.multiply(mat2, mat3);

    // Expected result:

    float expected[2*3] = {
        3, 4, 5,
        9, 14, 19
    };

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (mat3.get(i, j) != expected[i * 3 + j]) {
                std::cout << "Error at (" 
                          << i << ", " << j << "): "
                          << mat3.get(i, j) 
                          << " != " 
                          << expected[i * 3 + j] 
                          << std::endl;
            }
            std::cout << mat3.get(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "M2D test completed." << std::endl;
}


void testMLinear() {
    MLinear<float> linear(2, 2);
    /*
        W is already transposed 
        (saved and loaded as transposed)
        W = [1, 2]
            [3, 4]
        b = [1, 2]
    */
    for (int i = 0; i < 4; ++i) {
        linear.weight_.m_data[i] = i + 1.f; 
        // Initialize weights
    }

    for (int i = 0; i < 2; ++i) {
        linear.bias_.m_data[i] = i + 1.f; 
        // Initialize biases
    }

    // Input vector
    Mat2D<float> input(1, 2);
    for (int i = 0; i < 2; ++i) {
        input.m_data[i] = i + 1.f;
    }

    /*
        Output vector = input * W + b
        [1,2] [1,2]  + [1, 2] =  [7, 10] + [1, 2] = [8, 12]
              [3,4]
    */

    Mat2D<float> output(1, 2);
    linear.forward(input, output);

    // Check output
    float expected[2] = {8.f, 12.f};
    for (int i = 0; i < 2; ++i) {
        if (output.get(0, i) != expected[i]) {
            std::cout << "Error at output index "
                      << i << ": "
                      << output.get(0, i)
                      << " != "
                      << expected[i] << std::endl;
        } 
    }
    std::cout << "MLinear test completed." << std::endl;
}

void testMEmbed() {
    MEmbed<float> embed(4, 3);
    for (int i = 0; i < 12; ++i) {
        embed.embeddings_.m_data[i] = i + 1.f;
        // Initialize embeddings
    }

    // Input indices
    std::vector<int> indices = {0, 2, 1};
    Mat2D<float> output(3, 3); // Output matrix
    embed.forward(indices, output);

    // Check output
    float expected[3][3] = {
        {1.f, 2.f, 3.f}, // Embedding for index 0
        {7.f, 8.f, 9.f}, // Embedding for index 2
        {4.f, 5.f, 6.f}  // Embedding for index 1
    };  
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (output.get(i, j) != expected[i][j]) {
                std::cout << "Error at output (" 
                          << i << ", " << j << "): "
                          << output.get(i, j) 
                          << " != " 
                          << expected[i][j] 
                          << std::endl;
                break; // Break on first error
            }
        }
    }
    std::cout << "MEmbed test completed." << std::endl;
}

void testSoftmax() {
    Mat2D<float> input(3, 3);
    
    /*
        input = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    */
    for (int i = 0; i < 9; ++i) {
        input.m_data[i] = i + 1.f; // Initialize input
    }

    float softmax_output[9]= {
        0.09003057f, 0.24472847f, 0.66524096f,
        0.09003057f, 0.24472847f, 0.66524096f,
        0.09003057f, 0.24472847f, 0.66524096f
    };

    Mat2D<float> output(3, 3);
    softmax(input, output);
    // Check output
    for (int i = 0; i < 9; ++i) {
        if (std::fabs(output.m_data[i] - softmax_output[i]) > epsilon) {
            std::cout << "Error at output index "
                      << i << ": "
                      << output.m_data[i]
                      << " != "
                      << softmax_output[i] << std::endl;
            break; // Break on first error
        }
    }
    std::cout << "Softmax test completed." << std::endl;
}

int main(int argc, char** argv) {
    testMat2D();
    testMLinear();
    testMEmbed();
    testSoftmax();
    return 0;
}