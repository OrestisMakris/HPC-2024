#!/bin/sh
# Optimization info -fopt-info
#gcc $1 -o $2 -fopenmp -lgomp -foffload=nvptx-none -lcuda $3 $4 $5 $6
gcc-14 $1 -o $2 -fopenmp -lgomp -foffload=nvptx-none -lcuda $3 $4 $5 $6 -foffload="-misa=sm_75" -fcf-protection=none -fno-stack-protector

