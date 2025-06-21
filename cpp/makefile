CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wfatal-errors

SRC = test_MTB.cpp
TARGET = ./build/test_MTB

all: $(TARGET)

$(TARGET): $(SRC) 
	@mkdir -p ./build
	$(CXX) $(CXXFLAGS) $(SRC) MTB.hpp -o $(TARGET)

clean:
	rm -rf ./build

.PHONY: all clean
