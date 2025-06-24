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
  --m 1 2 2 \
  --n 2 2 \
  --save_path ${SAVE_PATH} \
  --dtype float32

./build/test/test_matmul_ext \
  --m 1 2 2 \
  --n 2 2 \
  --path ${SAVE_PATH}
echo " Finished case 1"

# case 2
echo " Starting case 2"
python  test/test_matmul_ext.py \
  --m 768 768 \
  --n 768 768 \
  --save_path ${SAVE_PATH} \
  --dtype float32   

./build/test/test_matmul_ext \
  --m 768 768 \
  --n 768 768 \
  --path ${SAVE_PATH}
echo " Finished case 2"