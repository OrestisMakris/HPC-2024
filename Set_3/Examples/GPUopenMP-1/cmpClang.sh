#!/bin/sh
# optimization info -Rpass=optimization -fsave-optimization-record 
clang -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_75 $1 -o $2 $3 $4 $5
