#!/bin/bash

set -ex

cmake .
make
mkdir models
python create_model.py --models_dir "models" --model_size "124M"
./gpt2
ctest

make clean
rm CMakeCache.txt
cmake -DFASTGPT_BLAS=OpenBLAS .
make
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2

rm model.dat
curl -O model.dat -L https://huggingface.co/datasets/certik/fastGPT/resolve/main/model_fastgpt_124M_v1.dat
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2

rm gpt2
python pt.py
