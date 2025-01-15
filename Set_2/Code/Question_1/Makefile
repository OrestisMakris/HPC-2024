# Makefile for CUDA program

# Compiler
NVCC = nvcc

# Compilation flags
NVCCFLAGS = -arch=sm_70 -O2

# Build rules
all: tiled_complex_mat_mul global_complex_mat_mul int_complex_mat_mul float_complex_mat_mul

tiled_complex_mat_mul: tiled_complex_mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

global_complex_mat_mul: global_complex_mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

int_complex_mat_mul: int_complex_mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

float_complex_mat_mul: float_complex_mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f tiled_complex_mat_mul global_complex_mat_mul int_complex_mat_mul float_complex_mat_mul

# Phony targets
.PHONY: all clean
