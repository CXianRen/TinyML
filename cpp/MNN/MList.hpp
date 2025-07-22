#pragma once

#include "Mmodel.hpp"
#include <memory>

namespace mnn {

template <typename T>
class MList: public MModel {
public:
std::vector<std::unique_ptr<T>> list_;

public:
    MList() {
        MACRO_CLASS_NAME(MList);
    }

    void add(std::unique_ptr<T> item) {
        item->name_ = std::to_string(list_.size());
        child_.push_back(item.get());
        list_.push_back(std::move(item));
    }

    std::unique_ptr<T>& operator[](size_t index) {
        return list_[index];
    }


    void printInfo(size_t indent = 0, 
      std::ostream& os = std::cout) const override {
        // if not the top model, print the name and type
        if(indent !=0){
          os << std::string(indent, ' ') 
                    << "(" << getModelName()  << ") :";
        }
          os << " " << getModelType() << " size:" << list_.size() << " (" << std::endl;

          if (!list_.empty()) {
            list_[0]->printInfo(indent + 2, os);
          }
          os << std::string(indent, ' ') << ")" << std::endl;
    }

    void loadParameters(const std::string& modelPath) override {
        for (auto& item : list_) {
            item->loadParameters(modelPath + "/" + item->getModelName());
        }
    }
};
}