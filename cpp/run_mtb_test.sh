#!/bin/bash 
set -e

./build/MTB/test/test_basic

./build/MTB/test/test_broadcast

./build/MTB/test/test_creator

./build/MTB/test/test_math

./build/MTB/test/test_matmul

bash ./MTB/test/test_matmul_ext.sh