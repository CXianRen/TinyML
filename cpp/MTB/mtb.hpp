#pragma once
// define everything here 

#include "tensor.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

#include <mtb_builder.hpp>
#include <mtb_bc.hpp>
#include <mtb_op_math.hpp>
#include <mtb_op_gemm.hpp>
