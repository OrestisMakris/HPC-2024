#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Every thread calculates one element of the resulting matrix
__global__ void mat_add(uint8_t* A, uint8_t* B, uint8_t* C, int N) {
    // Calculate the row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to avoid out-of-bounds access
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// Function to initialize matrix with random values
void initialize_matrix(uint8_t* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 10; // Random integers between 0 and 9
    }
}

// Verify the result on the CPU
void verify_add(uint8_t* A, uint8_t* B, uint8_t* C_cpu, uint8_t* C_gpu, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_cpu[i * N + j] = A[i * N + j] + B[i * N + j];
            // Compare the results
            if (C_gpu[i * N + j] != C_cpu[i * N + j]) {
                printf("Mismatch at index (%d, %d): %d != %d\n", i, j, C_gpu[i * N + j], C_cpu[i * N + j]);
                return;
            }
        }
    }
    printf("Addition is correct\n");
}

int main(void) {
    // Set square matrix dimension (NxN)
    int const N = 1024;
    size_t size = N * N * sizeof(uint8_t);

    // Allocate host memory
    uint8_t* A = (uint8_t*)malloc(size);
    uint8_t* B = (uint8_t*)malloc(size);
    uint8_t* C_cpu = (uint8_t*)malloc(size);
    uint8_t* C_gpu = (uint8_t*)malloc(size);

    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    // Allocate device memory
    uint8_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set Block and Grid dimensions
    int threads_per_block = 16;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    dim3 THREADS(threads_per_block, threads_per_block);
    dim3 BLOCKS(blocks, blocks);

    // Launch kernel
    mat_add<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Ensure all threads finish before proceeding

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    verify_add(A, B, C_cpu, C_gpu, N);

    // Print the first 5 elements of the resulting matrices
    printf("First 5 elements of A, B, and the resulting matrices (C_cpu, C_gpu):\n");
    for (int i = 0; i < 5; i++) {
        printf("A[%d] = %d, B[%d] = %d, C_cpu[%d] = %d, C_gpu[%d] = %d\n",
               i, A[i], i, B[i], i, C_cpu[i], i, C_gpu[i]);
    }

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
