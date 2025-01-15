#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;

    // Get the current device
    cudaGetDevice(&device);

    // Get device properties
    cudaGetDeviceProperties(&prop, device);

    // Print useful properties
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Maximum Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum Thread Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock Rate: %d kHz\n", prop.clockRate);
    printf("Total Constant Memory: %zu bytes\n", prop.totalConstMem);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);

    return 0;
}