#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

template <typename FP_TYPE>
std::vector<FP_TYPE> load_data(const std::string& filename, 
    size_t size, bool quiet = false) {
    std::ifstream file(filename, std::ios::binary| std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File not found");
    }
    // get file size
    size_t file_size = file.tellg(); 
    // back to the beginning
    file.seekg(0, std::ios::beg);
    if (file_size <= 0) {
        std::cerr << 
            "Error: File is empty or could not determine size: " 
            << filename << std::endl;
        throw std::runtime_error(
            "File is empty or could not determine size");
    }
    // check if file size is a multiple of FP_TYPE size
    if (file_size % sizeof(FP_TYPE) != 0) {
        throw std::runtime_error(
            "File size is not a multiple of FP_TYPE size");
    }
  
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

    if (!quiet) {
        std::cout << "Loaded " << size 
                  << " elements from file: \n\t" 
                  << filename << std::endl;
    }
    
    return data; 
}