/*Complex Matrix multiplication on GPU using shared memory*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"

#define THREADS_PER_BLOCK 16
#define TILE_SIZE THREADS_PER_BLOCK


__global__ void compute_E(int* A, int* B, int* C, int* D, int* E, int N){
    // Allocate shared memory 
    __shared__ int s_A[TILE_SIZE][TILE_SIZE];
    __shared__ int s_B[TILE_SIZE][TILE_SIZE];
    __shared__ int s_C[TILE_SIZE][TILE_SIZE];
    __shared__ int s_D[TILE_SIZE][TILE_SIZE];

    // Calculate the global row and column for each thread 
    // The row and column indices determine which element of the output matrix E the thread will compute.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    int sum_AC = 0;
    int sum_BD = 0;

    // Move the tile across the length of the grid 
    for(int i = 0; i < (N + dim - 1) / dim; i++){ // Handle non-multiples of dim
        // Load dim x dim tiles into shared memory, with boundary checks
        if(row < N && (i * dim + tx) < N){
            s_A[ty][tx] = A[row * N + i * dim + tx];
            s_B[ty][tx] = B[row * N + i * dim + tx];
        } else {
            s_A[ty][tx] = 0;
            s_B[ty][tx] = 0;
        }
        if(col < N && (i * dim + ty) < N){
            s_C[ty][tx] = C[(i * dim + ty) * N + col];
            s_D[ty][tx] = D[(i * dim + ty) * N + col];
        } else {
            s_C[ty][tx] = 0;
            s_D[ty][tx] = 0;
        }
        __syncthreads(); // Ensures all threads have completed loading their respective tiles

        // Accumulate the dot products
        for(int j = 0; j < dim; j++){
            sum_AC += s_A[ty][j] * s_C[j][tx];
            sum_BD += s_B[ty][j] * s_D[j][tx];
        }
        __syncthreads(); // Ensures all threads have completed the dot products before loading the next tile
    }
    // Write back the subtraction to E
    if(row < N && col < N){
        E[row * N + col] = sum_AC - sum_BD;
    }
}

__global__ void compute_F(int* A, int* B, int* C, int* D, int* F, int N) {
    // Allocate shared memory 
    __shared__ int s_A[TILE_SIZE][TILE_SIZE];
    __shared__ int s_B[TILE_SIZE][TILE_SIZE];
    __shared__ int s_C[TILE_SIZE][TILE_SIZE];
    __shared__ int s_D[TILE_SIZE][TILE_SIZE];

    // Calculate the global row and column for each thread 
    // The row and column indices determine which element of the output matrix F the thread will compute.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    int sum_AD = 0;
    int sum_BC = 0;

    for (int i = 0; i < (N + dim - 1) / dim; i++) {
        // Load tiles into shared memory, with boundary checks
        if (row < N && (i * dim + tx) < N) {
            s_A[ty][tx] = A[row * N + i * dim + tx];
            s_B[ty][tx] = B[row * N + i * dim + tx];
        } else {
            s_A[ty][tx] = 0;
            s_B[ty][tx] = 0;
        }
        if (col < N && (i * dim + ty) < N) {
            s_C[ty][tx] = C[(i * dim + ty) * N + col];
            s_D[ty][tx] = D[(i * dim + ty) * N + col];
        } else {
            s_C[ty][tx] = 0;
            s_D[ty][tx] = 0;
        }
        __syncthreads();

        // Accumulate the dot products
        for (int j = 0; j < dim; j++) {
            sum_AD += s_A[ty][j] * s_D[j][tx];
            sum_BC += s_B[ty][j] * s_C[j][tx];
        }
        __syncthreads();
    }

    // Write back the sum to F
    if (row < N && col < N) {
        F[row * N + col] = sum_AD + sum_BC;
    }
}

int main(int argc, char *argv[]){
    if (argc != 2) {
            printf("Usage: %s <matrix_dimension>\n", argv[0]);
            return 1;
        }

    // Set square matrix dimension (NxN)
    int N = atoi(argv[1]);

    size_t size = N * N * sizeof(int);

    // Allocate host memory
    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C = (int *)malloc(size);
    int *D = (int *)malloc(size);
    int *E_cpu = (int *)malloc(size);
    int *E_gpu = (int *)malloc(size);
    int *F_cpu = (int *)malloc(size);
    int *F_gpu = (int *)malloc(size);

    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);

    // Print the matrices
    /*print_matrix(A, N, "Matrix A");
    print_matrix(B, N, "Matrix B");
    print_matrix(C, N, "Matrix C");
    print_matrix(D, N, "Matrix D");*/

    // Allocate device memory
    int *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
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

    // blocks = (N + threads - 1) / threads, instead of N / threads 
    // to ensure all elements are covered if N is not a multiple of threads
    int blocks = (N + threads - 1) / threads;

    // Set up kernel launch parameters
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
    printf("GPU Execution time using shared memory and integer values: %f ms\n", execution_time);

    // Copy result back to host
    cudaMemcpy(E_gpu, d_E, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(F_gpu, d_F, size, cudaMemcpyDeviceToHost);

    // Print E_gpu and F_gpu for debugging
    //print_matrix(E_gpu, N, "Matrix E_gpu");
    //print_matrix(F_gpu, N, "Matrix F_gpu");


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