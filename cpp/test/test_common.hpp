#ifndef __TEST_COMMON_HPP__
#define __TEST_COMMON_HPP__
#include "MTB.hpp"

// compare two vectors for equality
template <typename T>
bool compare_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}


#endif // __TEST_COMMON_HPP__