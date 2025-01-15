/*Matrix multiplication on GPU using shared memory*/

# include <cuda.h>
# include <stdio.h>
# include <stdlib.h>

#define THREADS_PER_BLOCK 16
#define TILE_SIZE THREADS_PER_BLOCK

// Every threads calculates one element of the resulting matrix
__global__ void mat_mul(int* A, int* B, int* C, int N){
    // Allocate shared memory 
    __shared__ int s_A[TILE_SIZE][TILE_SIZE];
    __shared__ int s_B[TILE_SIZE][TILE_SIZE];

    // Calculate the global row and column for each thread 
    // The row and column indices determine which element of the output matrix C the thread will compute.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    int temp = 0;

    // Move the tile across the length of the grid 
    for(int i = 0; i < (N + dim - 1) / dim; i++){ // Handle non-multiples of dim
        // Load elements into shared memory, with boundary checks
        if(row < N && (i * dim + tx) < N){
            s_A[ty][tx] = A[row * N + i * dim + tx];
        } else {
            s_A[ty][tx] = 0;
        }
        if(col < N && (i * dim + ty) < N){
            s_B[ty][tx] = B[(i * dim + ty) * N + col];
        } else {
            s_B[ty][tx] = 0;
        }
        __syncthreads(); // Ensures all threads have completed loading their respective tiles

        // Accumulate the result
        for(int j = 0; j < dim; j++){
            temp += s_A[ty][j] * s_B[j][tx];
        }
        __syncthreads(); // Ensures all threads have completed the computation
    }
    // Write back the result to the output matrix
    if(row < N && col < N){
        C[row * N + col] = temp;
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
            C_cpu[i * N + j] = 0;
            for(int k = 0; k < N; k++){
                C_cpu[i * N + j] += A[i * N + k] * B[k * N + j];
            }
            // Compare the results
            if(C_gpu[i * N + j] != C_cpu[i * N + j]){
                printf("Mismatch at index %d, %d: %d != %d\n", i, j, C_cpu[i * N + j], C_gpu[i * N + j]);
                return;
            }
        }
    }
    printf("Multiplication is correct!\n");
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
    mat_mul<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); 

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    verify_mul(A, B, C_cpu, C_gpu, N);

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