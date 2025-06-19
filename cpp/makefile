CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

SRC = test_base.cpp
TARGET = ./build/test_base

all: $(TARGET)

$(TARGET): $(SRC) 
	@mkdir -p ./build
	$(CXX) $(CXXFLAGS) $(SRC) MNN.hpp -o $(TARGET)

clean:
	rm -rf ./build

.PHONY: all clean
