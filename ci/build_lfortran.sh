#!/bin/bash

set -ex

patch -p1 < ci/lfortran_namelist.patch

curl -o model.gguf -L https://huggingface.co/certik/fastGPT/resolve/main/model_fastgpt_124M_v2.gguf

mkdir lf
cd lf
FC=lfortran CMAKE_PREFIX_PATH=$CONDA_PREFIX cmake -DFASTGPT_BLAS=OpenBLAS -DCMAKE_BUILD_TYPE=Debug ..
make VERBOSE=1
ln -s ../model.gguf .
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./test_basic_input
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./test_more_inputs
cd ..

mkdir lf-fast
cd lf-fast
FC="lfortran --fast" CMAKE_PREFIX_PATH=$CONDA_PREFIX cmake -DFASTGPT_BLAS=OpenBLAS -DCMAKE_BUILD_TYPE=Release ..
make VERBOSE=1
ln -s ../model.gguf .
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./gpt2
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./test_basic_input
time OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./test_more_inputs
cd ..
