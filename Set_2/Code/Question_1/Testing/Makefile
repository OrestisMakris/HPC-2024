# Makefile for CUDA program

# Compiler
NVCC = nvcc

# Compilation flags
NVCCFLAGS = -arch=sm_70 -O2

# Build rules
all: add_vec device_properties mat_mul mat_mul_add mat_mul_opt shared_mat_mul shared_mat_add

add_vec: add_vec.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

device_properties: device_properties.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mat_mul: mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mat_add: mat_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mat_mul_opt: mat_mul_opt.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

shared_mat_mul: shared_mat_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

shared_mat_add: shared_mat_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f add_vec device_properties mat_mul mat_add mat_mul_opt shared_mat_mul shared_mat_add

# Phony targets
.PHONY: all clean
