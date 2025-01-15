# include <cuda.h>
# include <stdio.h>
# include <stdlib.h>

// Every threads calculates one element of the resulting matrix
__global__ void mat_mul(int* A, int* B, int* C, int N){
    // Calculate the row and column for each thread 
    // row is in the y dim and col is in the x dim
    // start + offset
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check, beacause it is possible 
    // for the number of threads to be greater than the number of elements in the matrix
    if(row < N && col < N){
        // Itirate over the row and column
        for(int i = 0; i < N; i++){
            C[row * N + col] += A[row * N + i] * B[i * N + col];
        }

    }

}

// Function to initialize matrix with random integer values
void initialize_matrix(int *matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = rand() % 10; // Random integers between 0 and 9
    }
}

// Verify the result on the CPU
void verify_mul(int* A, int* B, int* C_cpu, int* C_gpu, int N){
    // CPU implementation of matrix multiplication
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C_cpu[i * N + j] += A[i * N + k] * B[k * N + j];
            }
            // Compare the results
            if(C_gpu[i * N + j] != C_cpu[i * N + j]){
                printf("Mismatch at index %d: %d != %d\n", i, C_gpu[i * N + j], C_cpu[i * N + j]);
                return;
            }
        }
    }
    printf("Multiplication is correct!\n");
}

int main(void){
// Set square matrix dimension (NxN) 
int const N = 5;
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
    int threads_per_block = 16;

    // blocks = (N + threads_per_block - 1) / threads_per_block, instead of N / threads_per_block 
    // to ensure all elements are covered if N is not a multiple of threads_per_block
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Set up kernel launch parameters
    dim3 THREADS(threads_per_block, threads_per_block);
    dim3 BLOCKS(blocks, blocks);

    // Launch kernel
    mat_mul<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Because kernel launches are asynchronous

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    verify_mul(A, B, C_cpu, C_gpu, N);

    // Print the first row of A, first col of B and the first entry of the resulting matrices C_cpu and C_gpu
    printf("First row of A: ");
    for(int i = 0; i < N; i++){
        printf("%d ", A[i]);
    }
    printf("\n");
    printf("First col of B: ");
    for(int i = 0; i < N; i++){
        printf("%d ", B[i * N]);
    }
    printf("\n");
    printf("First element of the resulting matrices (C_cpu, C_gpu): %d, %d\n", C_cpu[0], C_gpu[0]); 

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