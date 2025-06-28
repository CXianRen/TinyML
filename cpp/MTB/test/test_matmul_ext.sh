#!/bin/bash
set -e
# check path exists
if [ ! -d "build/MTB/test/" ]; then
  echo "Directory ../build/MTB/test/ does not exist."
  echo "Current directory: $(pwd)"
  exit 1
fi

SAVE_PATH=build/test/temp/
mkdir -p ${SAVE_PATH}

PY_SCRIPT_PATH=$(dirname "$0")/test_matmul_ext.py
BIN_PATH=./build/MTB/test/test_matmul_ext

# case 1
echo " Starting case 1"
python  ${PY_SCRIPT_PATH} \
  --m 2 2 \
  --n 2 2 \
  --save_path ${SAVE_PATH} \
  --dtype float32

${BIN_PATH} \
  --m 2 2 \
  --n 2 2 \
  --path ${SAVE_PATH}
echo " Finished case 1"

# case 2
echo " Starting case 2"
python  ${PY_SCRIPT_PATH} \
  --m 128 128 \
  --n 128 128 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

${BIN_PATH} \
  --m 128 128 \
  --n 128 128 \
  --path ${SAVE_PATH}
echo " Finished case 2"

# case 3
echo " Starting case 3"
python  ${PY_SCRIPT_PATH} \
  --m 1 5 128 128 \
  --n 128 128 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

${BIN_PATH} \
  --m 1 5 128 128 \
  --n 128 128 \
  --path ${SAVE_PATH}
echo " Finished case 3"


# case 4
echo " Starting case 4"
python  ${PY_SCRIPT_PATH} \
  --m 1 1024 1024 \
  --n 1024 1024 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

${BIN_PATH} \
  --m 1 1024 1024 \
  --n 1024 1024 \
  --path ${SAVE_PATH}
echo " Finished case 4"


echo " Starting case 5"
python  ${PY_SCRIPT_PATH} \
  --m 1 16 5 64 \
  --n 1 16 64 5 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

${BIN_PATH} \
  --m 1 16 5 64 \
  --n 1 16 64 5 \
  --path ${SAVE_PATH}
echo " Finished case 5"

echo " Starting case 6"
# 1 5 16 48 -> 1 16 5 48 (0, 2, 1, 3)
# 1 5 16 48 -> 1 16 5 48 -> 1 16 48 5 (0, 2, 3, 1)
python  ${PY_SCRIPT_PATH} \
  --m 1 5 16 48 \
  --n 1 5 16 48 \
  --mt 0 2 1 3 \
  --nt 0 2 3 1 \
  --save_path ${SAVE_PATH} \
  --dtype float32

${BIN_PATH} \
  --m 1 5 16 48 \
  --n 1 5 16 48 \
  --mt 0 2 1 3 \
  --nt 0 2 3 1 \
  --path ${SAVE_PATH}
echo " Finished case 6"

