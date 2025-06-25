SUBDIRS := MNN MTB

.PHONY: all clean $(SUBDIRS) $(addsuffix _test, $(SUBDIRS))

all: $(SUBDIRS)

BUILD_DIR := build

# compile a specific module
$(SUBDIRS):
	echo "Compiling $@..."
	$(MAKE) -C $@ BUILD_DIR=$(BUILD_DIR)

# compile the test of a specific module
%_test:
	$(MAKE) -C $* test BUILD_DIR=$(BUILD_DIR)

# clean all modules
clean:
	rm -rf $(BUILD_DIR)
