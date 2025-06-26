#ifndef __TEST_COMMON_HPP__
#define __TEST_COMMON_HPP__

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include "mtb.hpp"


#define START_TEST() \
    std::cout << \
    "Start " \
    << __FUNCTION__ << std::endl


#define PASSLOG() \
    std::cout << \
    "[Passed] " \
    << __FUNCTION__ << std::endl

template <typename FP_TYPE>
std::vector<FP_TYPE> load_data(const std::string& filename, int size) {
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

template <typename FP_TYPE>
void compare_data(FP_TYPE* data1, FP_TYPE* data2, int size, 
                  double threshold = 1e-5) {
    double max_error = 0.0;
    double average_error = 0.0;
    for (int i = 0; i < size; ++i) {
        double error = std::abs(data1[i] - data2[i]);
        max_error = std::max(max_error, error);
        average_error += error;
        if (error > 0.00001*std::abs(data2[i])+ 1e-5) {
            std::cerr << "Data mismatch at index " << i 
                      << ": " << data1[i] << " != " << data2[i] << std::endl;
            throw std::runtime_error("Data mismatch");
        }
    }
    average_error /= size;
    std::cout << "Max error: " << max_error 
              << ", Average error: " << average_error << std::endl;
}

#endif // __TEST_COMMON_HPP__