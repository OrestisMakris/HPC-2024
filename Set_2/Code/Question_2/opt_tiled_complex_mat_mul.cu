/*
Complex Matrix multiplication on GPU using shared memory
(AC - BD) + (BC + AD)i = E + Fi,

With the following optimization:
Instead of:
    E = AC - BD
    F = AD + BC
    Resulting in 2 matrix additions and 4 matrix multiplications.

We can compute:
    E = Q + R
    F = P - Q
    Where:
    P = (A+B)C
    Q = A(C-D)
    R = (A-B)D
    Resulting in 5 matrix additions and 3 matrix multiplications, because Q is computed once and used twice.
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"

#define THREADS_PER_BLOCK 16 
#define TILE_SIZE THREADS_PER_BLOCK

__global__ void compute_PQR(int* A, int* B, int* C, int* D, int* P, int* Q, int* R, int N){
    __shared__ int s_A[TILE_SIZE][TILE_SIZE];
    __shared__ int s_B[TILE_SIZE][TILE_SIZE];
    __shared__ int s_C[TILE_SIZE][TILE_SIZE];
    __shared__ int s_D[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int dim = blockDim.x;
    
    int sum_P = 0, sum_Q = 0, sum_R = 0;
    
    for (int i = 0; i < (N + dim - 1) / dim; i++) {
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
        
        for (int j = 0; j < dim; j++) {
            sum_P += (s_A[ty][j] + s_B[ty][j]) * s_C[j][tx];
            sum_Q += s_A[ty][j] * (s_C[j][tx] - s_D[j][tx]);
            sum_R += (s_A[ty][j] - s_B[ty][j]) * s_D[j][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        P[row * N + col] = sum_P;
        Q[row * N + col] = sum_Q;
        R[row * N + col] = sum_R;
    }
}

__global__ void compute_EF(int* P, int* Q, int* R, int* E, int* F, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int idx = row * N + col;
        E[idx] = Q[idx] + R[idx];
        F[idx] = P[idx] - Q[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(int);

    // Allocate host memory 
    int *A, *B, *C, *D, *E_cpu, *E_gpu, *F_cpu, *F_gpu;
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    D = (int *)malloc(size);
    E_cpu = (int *)malloc(size);
    E_gpu = (int *)malloc(size);
    F_cpu = (int *)malloc(size);
    F_gpu = (int *)malloc(size);

    initialize_matrix(A, N);
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);

    // Allocate device memory 
    int *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_P, *d_Q, *d_R;
    cudaMalloc(&d_A, size); 
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_D, size); 
    cudaMalloc(&d_E, size); 
    cudaMalloc(&d_F, size);
    cudaMalloc(&d_P, size); 
    cudaMalloc(&d_Q, size); 
    cudaMalloc(&d_R, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;
    dim3 THREADS(threads, threads), BLOCKS(blocks, blocks);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0);

    compute_PQR<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, d_D, d_P, d_Q, d_R, N);
    cudaDeviceSynchronize(); // Ensure P, Q, R are computed before computing E, F
    compute_EF<<<BLOCKS, THREADS>>>(d_P, d_Q, d_R, d_E, d_F, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("GPU Execution time with optimized complex matrix multiplication: %f ms\n", execution_time);
    
    cudaMemcpy(E_gpu, d_E, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(F_gpu, d_F, size, cudaMemcpyDeviceToHost);
    
    verify_complex_mat_mul(A, B, C, D, E_cpu, E_gpu, F_cpu, F_gpu, N);
    
    free(A); free(B); free(C); free(D); free(E_cpu); free(E_gpu); free(F_cpu); free(F_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_E); cudaFree(d_F); cudaFree(d_P); cudaFree(d_Q); cudaFree(d_R);
    return 0;
}