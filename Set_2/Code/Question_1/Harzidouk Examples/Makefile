all: hello1 vector_addition1 red00 red01 cpi vector_addition2

NVFLAGS = --gpu-architecture=sm_35

hello1: hello1.cu
	nvcc $(NVFLAGS) -o hello1 hello1.cu

vector_addition1: vector_addition1.cu
	nvcc $(NVFLAGS) -o vector_addition1 vector_addition1.cu

vector_addition2: vector_addition2.cu
	nvcc $(NVFLAGS) -o vector_addition2 vector_addition2.cu

red00: red00.cu
	nvcc $(NVFLAGS) -o red00 red00.cu

red01: red01.cu
	nvcc $(NVFLAGS) -o red01 red01.cu

cpi: cpi.cu
	nvcc $(NVFLAGS) -o cpi cpi.cu


clean:
	rm -f hello1 vector_addition1 vector_addition2 red00 red01 cpi