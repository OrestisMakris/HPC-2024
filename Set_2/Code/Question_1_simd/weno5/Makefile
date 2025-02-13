CC=gcc
CFLAGS= -Wall -march=native -Wno-unused-value -O2 -fno-tree-vectorize

# Auto-vectorization flags
CFLAGS_AUTO= -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize  -ftree-vectorizer-verbose=5 -fopt-info-vec

# builtin flags
CFLAGS_BUILTIN= -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize  -ftree-vectorizer-verbose=5 #-ffast-math -mfma

# OpenMP flags
CFLAGS_OMP = -O3 -fopenmp -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize -fopt-info -ffast-math

# OpenMP optim flags
CFLAGS_OMP_OPTIM = -O3 -fopenmp -Wall -march=native -Wno-unused-value -O3 -ftree-vectorize -fopt-info -ffast-math

# Vectorization SIMD flags for AVX implementation
CFLAGS_AVX= -O3 -mavx -msse4 -mfma -lm -Wno-unused-result -fopt-info-vec -march=native -Wall 

# Vectorization SIMD flags for SSE implementation
CFLAGS_SSE= -O3 -msse4 -lm -Wno-unused-result -fopt-info-vec -march=native -Wall 

#Vectorization SIMD flags for AVX FMA implementation
CFLAGS_AVX_FMA= -O3 -mavx -mfma -lm -Wno-unused-result -fopt-info-vec -march=native -Wall

#Vectorization SIMD flags for AVX Memory alignment implementation
CFLAGS_AVX_MEM_ALIGN= -O3 -mavx -mfma -lm -Wno-unused-result -fopt-info-vec -march=native -Wall

LIBS=-lm

all: bench bench_auto bench_builtin bench_omp bench_omp_optim bench_avx bench_sse bench_avx_fma bench_avx_aligned

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

bench_avx: bench.c weno.h
	$(CC) $(CFLAGS_AVX) -o bench_avx bench.c $(LIBS)

bench_sse: bench.c weno.h
	$(CC) $(CFLAGS_SSE) -msse4 -o bench_sse bench.c $(LIBS)

bench_avx_fma: bench.c weno.h
	$(CC) $(CFLAGS_AVX_FMA) -o bench_avx_fma bench.c $(LIBS)

bench_avx_aligned: bench.c weno.h
	$(CC) $(CFLAGS_AVX_MEM_ALIGN) -o bench_avx_aligned bench.c $(LIBS)

clean:
	rm -f bench bench_auto bench_builtin bench_omp bench_omp_optim bench_avx bench_sse bench_avx_fma bench_avx_aligned 