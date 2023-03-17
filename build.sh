#!/bin/bash

set -ex

FC=gfortran cmake -Bbuild
cmake --build build --parallel
ctest --test-dir build -V
