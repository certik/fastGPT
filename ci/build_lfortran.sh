#!/bin/bash

set -ex

curl -o model.dat -L https://huggingface.co/datasets/certik/fastGPT/resolve/main/model_fastgpt_124M_v1.dat

mkdir lf
cd lf
FC=lfortran cmake -DFASTGPT_BLAS=OpenBLAS -DCMAKE_BUILD_TYPE=Debug ..
make VERBOSE=1
ln -s ../model.dat .
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2
cd ..

mkdir lf-fast
cd lf-fast
FC="lfortran --fast" cmake -DFASTGPT_BLAS=OpenBLAS -DCMAKE_BUILD_TYPE=Release ..
make VERBOSE=1 gpt2
ln -s ../model.dat .
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2
cd ..
