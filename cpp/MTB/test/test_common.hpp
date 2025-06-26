#ifndef __TEST_COMMON_HPP__
#define __TEST_COMMON_HPP__
#include "mtb.hpp"
#include <assert.h>
// compare two vectors for equality
template <typename T>
bool compare_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

#define START_TEST() \
    std::cout << \
    "Start " \
    << __FUNCTION__ << std::endl


#define PASSLOG() \
    std::cout << \
    "[Passed] " \
    << __FUNCTION__ << std::endl



#endif // __TEST_COMMON_HPP__