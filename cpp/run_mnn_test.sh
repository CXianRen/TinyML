#!/bin/bash 
set -e
mkdir -p build/test/temp

python MNN/test/test_linear.py
./build/MNN/test/test_linear

python MNN/test/test_embed.py
./build/MNN/test/test_embed

python MNN/test/test_selfAT.py
./build/MNN/test/test_selfAT

python MNN/test/test_selfAT_wh.py
./build/MNN/test/test_selfAT_wh

python MNN/test/test_gelu.py
./build/MNN/test/test_gelu

python MNN/test/test_layernorm.py
./build/MNN/test/test_layernorm