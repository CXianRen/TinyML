#pragma once

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include "mtb.hpp"
#include "common.hpp"

#define START_TEST() \
    std::cout << \
    "Start " \
    << __FUNCTION__ << std::endl


#define PASSLOG() \
    std::cout << \
    "[Passed] " \
    << __FUNCTION__ << std::endl

template <typename FP_TYPE>
void compare_data(FP_TYPE* data1, FP_TYPE* data2, int size, 
                  double threshold = 1e-5) {
    double max_error = 0.0;
    double average_error = 0.0;
    for (int i = 0; i < size; ++i) {
        double error = std::abs(data1[i] - data2[i]);
        max_error = std::max(max_error, error);
        average_error += error;
        if (error > 0.00001*std::abs(data2[i]) + threshold) {
            std::cerr << "Data mismatch at index " << i 
                      << ": " << data1[i] << " != " << data2[i]
                      << " (error: " << error << ")"
                      << ", threshold: " << threshold 
                      << std::endl;
            throw std::runtime_error("Data mismatch");
        }
    }
    if (max_error > 1e-2){
        std::cerr << "Max error is too large: " << max_error 
                  << ", threshold: " << threshold << std::endl;
        throw std::runtime_error("Max error is too large");
    }
    
    average_error /= size;
    std::cout << "Max error: " << max_error 
              << ", Average error: " << average_error << std::endl;
}
