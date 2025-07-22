#pragma once
#include <string>
#include <iostream>
#include <vector>

#include "mtb.hpp"

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

    virtual void printInfo(size_t indent = 0, 
        std::ostream& os = std::cout ) const {
      // if not the top model, print the name and type
      if(indent !=0){
        os << std::string(indent, ' ') 
                  << "(" << name_ << ") :";
      }
        os << " " << type_ << "(" << std::endl;

      for(const auto& child : child_) {
          child->printInfo(indent + 2, os);
      }
        os << std::string(indent, ' ') << ")" << std::endl;
    }
};
}