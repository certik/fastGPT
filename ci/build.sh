#!/bin/bash

set -ex

cmake -Bbuild
cmake --build build --parallel
ctest --test-dir build -V

cmake -DFASTGPT_BLAS=OpenBLAS -Bbuild
cmake --build build --parallel --clean-first
ctest --test-dir build -V

rm build/gpt2
python pt.py
