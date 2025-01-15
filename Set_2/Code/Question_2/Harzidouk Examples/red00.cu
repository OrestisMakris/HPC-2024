#include <stdio.h>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef unsigned long long Count;

const unsigned long WARP_SIZE = 1; // Warp size
const unsigned long NBLOCKS = 32; // Number of total cuda cores on my GPU
const unsigned long NSIZE = NBLOCKS*10000; // Number of points to generate (each thread)

// This kernel is
__global__ void reduce00(unsigned long *data, unsigned long *totals, unsigned long nsize) {
	// Unique ID of the thread
	int bid = blockIdx.x;
  int nblocks = gridDim.x;

  unsigned long chunk = nsize / nblocks;
  unsigned long start = chunk * bid;
  unsigned long end = start + chunk;

  //printf("b %d/%d: %lld %lld %lld\n", bid, nblocks, start, end, chunk);

	// Initialize the counter
	unsigned long counter = 0;

	// Computation loop
	for (unsigned long i = start; i < end; i++)
	{
		counter += data[i];
		//counter++;
	}

  //printf("b %d: %lld %lld -> %lld\n", bid, start, end, counter);
	totals[bid] = counter;
}

int main(int argc, char **argv) {
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		printf("CUDA device missing! Do you need to use optirun?\n");
		return 1;
	}

	printf("Starting simulation with %ld blocks, %ld threads/block, and %ld elements\n", NBLOCKS, WARP_SIZE, NSIZE);

	// Allocate host and device memory to store the counters
	unsigned long *hOut, *dOut;
	unsigned long *hData, *dData;
	hOut = (unsigned long *)malloc(sizeof(unsigned long)*NBLOCKS); // Host memory
	cudaMalloc(&dOut, sizeof(unsigned long) * NBLOCKS); // Device memory
	hData = (unsigned long *)malloc(sizeof(unsigned long)*NSIZE); // Host memory
	cudaMalloc(&dData, sizeof(unsigned long) * NSIZE); // Device memory

  for (unsigned long i = 0; i < NSIZE; i++) hData[i] = i+1;
	cudaMemcpy(dData, hData, sizeof(unsigned long) * NSIZE, cudaMemcpyHostToDevice);

  for (unsigned long i = 0; i < NBLOCKS; i++) hOut[i] = 0;
	cudaMemcpy(dOut, hOut, sizeof(unsigned long) * NBLOCKS, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventRecord(start,0);

	// Launch kernel
	reduce00<<<NBLOCKS, WARP_SIZE>>>(dData, dOut, NSIZE);

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime);

  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  //cudaDeviceSynchronize();

	// Copy back memory used on device and free
	cudaMemcpy(hOut, dOut, sizeof(unsigned long) * NBLOCKS, cudaMemcpyDeviceToHost);
	cudaFree(dOut);
	cudaFree(dData);

	// Compute total hits
	unsigned long total = 0;
	for (int i = 0; i < NBLOCKS; i++) {
		total += hOut[i];
	}
  printf("Total sum = %ld\n", total);

	return 0;
}