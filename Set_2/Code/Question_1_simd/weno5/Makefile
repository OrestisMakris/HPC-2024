CC=gcc
CFLAGS= -Wall -march=native -Wno-unused-value -O2 -fno-tree-vectorize

# Auto-vectorization flags
CFLAGS_AUTO= -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize  -ftree-vectorizer-verbose=5 -fopt-info-vec

# builtin flags
CFLAGS_BUILTIN= -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize  -ftree-vectorizer-verbose=5 #-ffast-math -mfma

# OpenMP flags
CFLAGS_OMP = -O3 -fopenmp -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize -fopt-info

# OpenMP optim flags
CFLAGS_OMP_OPTIM = -O3 -fopenmp -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize -fopt-info -ffast-math

LIBS=-lm

all: bench bench_auto bench_builtin bench_omp bench_omp_optim

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

bench_auto: bench.c weno.h
	$(CC) $(CFLAGS_AUTO) -o bench_auto bench.c $(LIBS)

bench_builtin: bench.c weno.h
	$(CC) $(CFLAGS_BUILTIN) -o bench_builtin bench.c $(LIBS)

bench_omp: bench.c weno.h
	$(CC) $(CFLAGS_OMP) -o bench_omp bench.c $(LIBS)

bench_omp_optim: bench.c weno.h
	$(CC) $(CFLAGS_OMP_OPTIM) -o bench_omp_optim bench.c $(LIBS)

clean:
	rm -f bench bench_auto bench_builtin bench_omp bench_omp_optim