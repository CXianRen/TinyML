CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wfatal-errors

TEST_SRC := $(wildcard test/test_*.cpp)
TARGETS := $(patsubst %.cpp,build/%,${TEST_SRC})

INCLUDE_DIRS := -I.
TEST_INCLUDE_DIRS := -I./test

all: $(TARGETS)

build/test/%: test/%.cpp MTB.hpp | build/test
	$(CXX) $(CXXFLAGS) \
	$(INCLUDE_DIRS) \
	$(TEST_INCLUDE_DIRS) $< -o $@

build/test:
	@mkdir -p build/test

clean:
	rm -rf build

.PHONY: all clean build/test
