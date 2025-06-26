#include "test_common.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace mtb;

#define FP_TYPE float
typedef Tensor<FP_TYPE> TensorF;

std::vector<FP_TYPE> load_data(const std::string& filename, size_t size) {
    std::ifstream file(filename, std::ios::binary| std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File not found");
    }
    // get file size
    std::streamsize file_size = file.tellg(); 
    // back to the beginning
    file.seekg(0, std::ios::beg);
    if (file_size <= 0) {
        std::cerr << "Error: File is empty or could not determine size: " 
                  << filename << std::endl;
        throw std::runtime_error("File is empty or could not determine size");
    }
    // check if file size is a multiple of FP_TYPE size
    if (file_size % sizeof(FP_TYPE) != 0) {
        throw std::runtime_error("File size is not a multiple of FP_TYPE size");
    }
    std::cout << "File size: " << file_size/sizeof(FP_TYPE) << " Float32" << std::endl;

    if(size >0){
        // check file size == size * sizeof(FP_TYPE)
        if (file_size != size * sizeof(FP_TYPE)) {
            throw std::runtime_error(
                "File size does not match expected size");
        }
    }
    size = file_size / sizeof(FP_TYPE);
    std::vector<FP_TYPE> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cerr << "Error reading file: " << filename << std::endl;
        throw std::runtime_error("Error reading file");
    }
    return data; 
}


int main(int argc, char** argv) {
    // parameters --m --n --path 
    if (argc < 4) {
        std::cerr << "Usage: " 
            << argv[0] 
            << " --m <i j k ...> (m shape) "
            << "--n <l m ...>(n shape) "
            << "--path <data_file_path>" 
            << std::endl;
        return 1;
    }

    std::vector<size_t> shape_m;
    std::vector<size_t> mt;
    std::vector<size_t> shape_n;
    std::vector<size_t> nt;
    std::string data_file_path;

    // Helper lambda to parse shape arguments
    auto parse_shape = [&](size_t& i, std::vector<size_t>& shape) {
        ++i;
        while (i < argc && argv[i][0] != '-') {
            shape.push_back(std::stoi(argv[i]));
            ++i;
        }
        --i;
    };

    // Parse command line arguments
    for (size_t i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--m") {
            parse_shape(i, shape_m);
        }
        else if (arg == "--mt"){
            parse_shape(i, mt);
        } 
        else if (arg == "--n") {
            parse_shape(i, shape_n);
        }
        else if (arg == "--nt"){
            parse_shape(i, nt);
        } else if (arg == "--path") {
            if (++i < argc) {
                data_file_path = argv[i];
            }
        }
    }

    std::cout << "Shape M: ";
    for (auto v : shape_m) std::cout << v << " ";
    std::cout << ", Shape N: ";
    for (auto v : shape_n) std::cout << v << " ";
    std::cout << std::endl;

    auto a_file = data_file_path + "/a.bin";
    auto b_file = data_file_path + "/b.bin";
    auto c_file = data_file_path + "/c.bin";

    auto a_data = load_data(a_file, 0);
    auto b_data = load_data(b_file, 0);
    auto c_data = load_data(c_file, 0);
    
    TensorF a(shape_m, a_data);
    TensorF b(shape_n, b_data);

    std::cout << "a shape: " << a.shape() << " strides: " 
              << a.strides() << std::endl;
    std::cout << "b shape: " << shape_n << " strides: " 
              <<  b.strides() << std::endl;
    
    if(mt.size() > 0){
        std::cout << "transpose a: " << mt << std::endl;
        a = mtb::transpose(a, mt);
    }
    if(nt.size() > 0){
        std::cout << "transpose b: " << nt << std::endl;
        b = mtb::transpose(b, nt);
    }

    TensorF c=mtb::matmul(a, b);

    // check if c data matches
    if (c.size() != c_data.size()) {
        std::cerr << "Error: Output size mismatch. Expected "
                  << c_data.size() 
                  << ", got " << c.size() << std::endl;
        return 1;
    }

    // check if c data matches
    auto c_ptr = c.data();
    double error_sum = 0.0;
    double max_error = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
       // compute the average error
        double error = std::abs(c_ptr[i] - c_data[i]);
        error_sum += error;
        if (error > max_error) {
            max_error = error;
        }
        if (error > 0.00001*std::abs(c_data[i])+ 1e-5) {
            std::cerr << "Error: Output data mismatch at index " 
                      << i << ": expected " << c_data[i] 
                      << ", got " << c_ptr[i] 
                      << " (error: " << error << ")" << std::endl;
            return 1;
        }
    }

    double average_error = error_sum / c.size();
    std::cout << "Average error: " << average_error << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    return 0;
}