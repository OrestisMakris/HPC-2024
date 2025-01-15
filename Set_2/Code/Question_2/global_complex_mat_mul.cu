/*Complex Matrix multiplication on GPU using global memory*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"

#define THREADS_PER_BLOCK 16

__global__ void compute_E(float* A, float* B, float* C, float* D, float* E, int N) {
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_AC = 0.0f;
    float sum_BD = 0.0f;

    // Perform computation directly from global memory
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum_AC += A[row * N + k] * C[k * N + col];
            sum_BD += B[row * N + k] * D[k * N + col];
        }
        E[row * N + col] = sum_AC - sum_BD;
    }
}

__global__ void compute_F(float* A, float* B, float* C, float* D, float* F, int N) {
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_AD = 0.0f;
    float sum_BC = 0.0f;

    // Perform computation directly from global memory
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum_AD += A[row * N + k] * D[k * N + col];
            sum_BC += B[row * N + k] * C[k * N + col];
        }
        F[row * N + col] = sum_AD + sum_BC;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }

    // Set square matrix dimension (NxN)
    int N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);
    float *D = (float *)malloc(size);
    float *E_cpu = (float *)malloc(size);
    float *E_gpu = (float *)malloc(size);
    float *F_cpu = (float *)malloc(size);
    float *F_gpu = (float *)malloc(size);

    // Initialize matrices
    initialize_matrix(A, N); // Ensure initialize_matrix is updated for float
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_D, size);
    cudaMalloc(&d_E, size);
    cudaMalloc(&d_F, size);

    // Copy matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);

    // Set Block and Grid dimensions
    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record execution time for both kernels
    cudaEventRecord(start, 0);

    compute_E<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, d_D, d_E, N);
    compute_F<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, d_D, d_F, N);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float execution_time = 0;
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("GPU Execution time using just global memory: %f ms\n", execution_time);

    // Copy result back to host
    cudaMemcpy(E_gpu, d_E, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(F_gpu, d_F, size, cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    verify_complex_mat_mul(A, B, C, D, E_cpu, E_gpu, F_cpu, F_gpu, N);

    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E_cpu);
    free(E_gpu);
    free(F_cpu);
    free(F_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_F);

    return 0;
}