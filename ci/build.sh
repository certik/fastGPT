#!/bin/bash

set -ex

cmake .
make
mkdir models
python create_model.py --models_dir "models" --model_size "124M"
python encode_input.py \
    "Alan Turing theorized that computers would one day become very powerful, but even he could not imagine" \
    -n 20
./gpt2

make clean
rm CMakeCache.txt
cmake -DFASTGPT_BLAS=OpenBLAS .
make
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2

rm gpt2
python pt.py
