#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 16
#define TILE_SIZE THREADS_PER_BLOCK

__global__ void mat_add(int *A, int *B, int *C, int N) {
    // Shared memory for the current block
    __shared__ int s_A[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ int s_B[THREADS_PER_BLOCK][THREADS_PER_BLOCK];

    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Load data into shared memory
    if (row < N && col < N) {
        s_A[ty][tx] = A[row * N + col];
        s_B[ty][tx] = B[row * N + col];
    } else {
        s_A[ty][tx] = 0;
        s_B[ty][tx] = 0;
    }
    __syncthreads();

    // Compute the addition using a temporary variable
    int temp = 0;
    if (row < N && col < N) {
        temp = s_A[ty][tx] + s_B[ty][tx];
        C[row * N + col] = temp; // Store the result in global memory
    }
    __syncthreads();
}

// Function to initialize matrix with random integer values
void initialize_matrix(int *matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = rand() % 10; // Random integers between 0 and 9
    }
}

// Verify the result on the CPU
void verify_add(int* A, int* B, int* C_cpu, int* C_gpu, int N){
    // CPU implementation of matrix addition
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            C_cpu[i * N + j] = A[i * N + j] + B[i * N + j];

            // Compare the results
            if(C_cpu[i * N + j] != C_gpu[i * N + j]){
                printf("Mismatch at index (%d, %d): CPU = %d, GPU = %d\n", i, j, C_cpu[i * N + j], C_gpu[i * N + j]);
                return;
            }
        }
    }
    printf("Addition is correct!\n");
}

int main(int argc, char **argv) {
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
    int *C_cpu = (int *)malloc(size);
    int *C_gpu = (int *)malloc(size);

    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

   // Set Block and Grid dimensions
    int threads = THREADS_PER_BLOCK;

    // blocks = (N + threads - 1) / threads, instead of N / threads 
    // to ensure all elements are covered if N is not a multiple of threads
    int blocks = (N + threads - 1) / threads;

    // Set up kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    
    // Launch kernel
    mat_add<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); 

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Validate the result
    verify_add(A, B, C_cpu, C_gpu, N);

    // Cleanup
    free(A); free(B); free(C_cpu); free(C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
