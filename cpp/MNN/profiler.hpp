#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <stack>

namespace MPF {

class PNode {
public:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    PNode* parent = nullptr;
    std::vector<PNode*> children;

    PNode(const std::string& name) : name(name) {
        start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end = std::chrono::high_resolution_clock::now();
    }

    long long duration_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    void print(size_t indent = 0, std::ostream& os = std::cout) const {
        std::string pad(indent, ' ');
        os << pad << name << ": " << duration_us() << " us\n";
        for (auto* child : children) {
            child->print(indent + 2, os);
        }
    }
};

inline std::stack<PNode*> gStack;
inline PNode* gRoot = nullptr;

inline void ProfilerStart(const std::string& name) {
    PNode* node = new PNode(name);
    if (gStack.empty()) {
        gRoot = node;
    } else {
        node->parent = gStack.top();
        gStack.top()->children.push_back(node);
    }
    gStack.push(node);
}

inline void ProfilerStop() {
    if (!gStack.empty()) {
        gStack.top()->stop();
        gStack.pop();
    }
}


inline void ProfilerPrint(std::ostream& os = std::cout) {
    if (gRoot) gRoot->print(0, os);
    else std::cerr << "No profile data.\n";
}


} // namespace MPF

#define MSTART(name) MPF::ProfilerStart(name)
#define MSTOP()      MPF::ProfilerStop()
#define MPRINT()     MPF::ProfilerPrint()
#define MPRINT_TO(os) MPF::ProfilerPrint(os)
