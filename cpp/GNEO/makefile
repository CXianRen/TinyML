CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wfatal-errors -fopenmp -march=native

# should be set by its' parent makefile
ifeq ($(BUILD_DIR),)
$(error BUILD_DIR is not set.)
endif

MODULE = GNEO

SRC_DIR = .

MTB_SRC = "../MTB/"
MNN_SRC = "../MNN/"

TEST_DIR = test
TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)

MODULE_BUILD_DIR = ../$(BUILD_DIR)/$(MODULE)
MODULE_TEST_BUILD_DIR = $(MODULE_BUILD_DIR)/$(TEST_DIR)

TEST_TARGETS := $(patsubst \
    $(TEST_DIR)/%.cpp, \
    $(MODULE_TEST_BUILD_DIR)/%, \
    $(TEST_SRC))

INCLUDE_DIRS := -I$(SRC_DIR) -I$(TEST_DIR) -I$(MTB_SRC) -I$(MNN_SRC) -I$(MODULE_BUILD_DIR)

MTB_HPP_FILES := $(wildcard $(MTB_SRC)/*.hpp)
MNN_HPP_FILES := $(wildcard $(MNN_SRC)/*.hpp)
GNEO_HPP_FILES := $(wildcard $(SRC_DIR)/*.hpp)
INCLUDE_FILES := $(MTB_HPP_FILES) $(MNN_HPP_FILES) $(GNEO_HPP_FILES)

all: test 

test: $(TEST_TARGETS)

$(MODULE_TEST_BUILD_DIR)/%: $(TEST_DIR)/%.cpp $(INCLUDE_FILES) | $(MODULE_TEST_BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $< -o $@

$(MODULE_TEST_BUILD_DIR):
	mkdir -p $(MODULE_TEST_BUILD_DIR)

clean:
	rm -rf $(MODULE_BUILD_DIR)

.PHONY: all clean test
