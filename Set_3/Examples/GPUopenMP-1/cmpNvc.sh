#!/bin/sh
# Diplay Optimizations -Minfo=all
nvcc -fopenmp -mp=gpu $1 -o $2 $3 $4 $5 $6
