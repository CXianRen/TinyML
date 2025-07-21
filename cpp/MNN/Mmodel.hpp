#pragma once
#include <string>
#include "mtb.hpp"
#include "vector"

#define MACRO_REGISTER_M_MEMEBR(name) \
    child_.push_back(&name);\
    name.name_ = #name

#define MACRO_CLASS_NAME(type) \
    type_ = #type

namespace mnn {

class MModel  {
public:
    std::string name_;
    std::string type_;
    std::vector<MModel*> child_;

public:
    MModel() = default;
    ~MModel() = default;

    // virtual bool loadModel(const std::string& modelPath) = 0;
    
    std::string getModelName() const {
        return name_;
    }

    std::string getModelType() const {
        return type_;
    }

    // mtb::Tensor<T> operator()(const mtb::Tensor<T>& input) {
    //     return forward(input);
    // }

    // virtual Tensor<T> forward(const Tensor<T>& input) = 0;

    virtual void printInfo(size_t indent = 0) const {
      // if not the top model, print the name and type
      if(indent !=0){
        std::cout << std::string(indent, ' ') 
                  << "(" << name_ << ") :";
      }
      std::cout << " " << type_ << "(" << std::endl;

      for(const auto& child : child_) {
          child->printInfo(indent + 2);
      }
      std::cout << std::string(indent, ' ') << ")" << std::endl;
    }
};
}