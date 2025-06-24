#!/bin/bash
set -e
# check path exists
if [ ! -d "build/test/" ]; then
  echo "Directory ../build/test/ does not exist."
  echo "Current directory: $(pwd)"
  exit 1
fi

SAVE_PATH=build/test/temp/
mkdir -p ${SAVE_PATH}


# case 1
echo " Starting case 1"
python  test/test_matmul_ext.py \
  --m 2 2 \
  --n 2 2 \
  --save_path ${SAVE_PATH} \
  --dtype float32

./build/test/test_matmul_ext \
  --m 2 2 \
  --n 2 2 \
  --path ${SAVE_PATH}
echo " Finished case 1"

# case 2
echo " Starting case 2"
python  test/test_matmul_ext.py \
  --m 128 128 \
  --n 128 128 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

./build/test/test_matmul_ext \
  --m 128 128 \
  --n 128 128 \
  --path ${SAVE_PATH}
echo " Finished case 2"

# case 3
echo " Starting case 3"
python  test/test_matmul_ext.py \
  --m 1 5 128 128 \
  --n 128 128 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

./build/test/test_matmul_ext \
  --m 1 5 128 128 \
  --n 128 128 \
  --path ${SAVE_PATH}
echo " Finished case 3"


# case 4
echo " Starting case 4"
python  test/test_matmul_ext.py \
  --m 1 1024 1024 \
  --n 1024 1024 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

./build/test/test_matmul_ext \
  --m 1 1024 1024 \
  --n 1024 1024 \
  --path ${SAVE_PATH}
echo " Finished case 4"