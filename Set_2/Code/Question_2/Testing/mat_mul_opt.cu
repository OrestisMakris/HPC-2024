/*Matrix multiplication on GPU using shared memory*/

# include <cuda.h>
# include <stdio.h>
# include <stdlib.h>

#define THREADS_PER_BLOCK 32
#define SM_SIZE (THREADS_PER_BLOCK * THREADS_PER_BLOCK) // Shared memory size

// Every threads calculates one element of the resulting matrix
__global__ void mat_mul(int* A, int* B, int* C, int N){
    // Allocate shared memory 
    __shared__ int s_A[SM_SIZE];
    __shared__ int s_B[SM_SIZE];

    // Calculate the global row and column for each thread 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    // Initiaize s_A and s_B to 0
    s_A[ty * dim + tx] = 0;
    s_B[ty * dim + tx] = 0;

    // print s_a and s_b
    /*printf("\n");
    printf("Block (%d, %d) Thread (%d, %d): s_A[%d] = %d, s_B[%d] = %d\n",
    blockIdx.y, blockIdx.x,
    threadIdx.y, threadIdx.x,
    ty * dim + tx, s_A[ty * dim + tx],
    ty * dim + tx, s_B[ty * dim + tx]);
    printf("\n");*/

    //if (row < N && col < N) { // Only process valid rows and columns
    C[row * N + col] = 0; // Initialize the result to 0

    // Move the tile across the length of the grid 
    for(int i = 0; i < (N + dim - 1) / dim; i++){ // Handle non-multiples of dim
        // Load elements into shared memory, with boundary checks
        if (((i * dim + tx) < N)) {
            s_A[ty * dim + tx] = A[(row * N) + (i * dim) + tx];
        }
        else{
            s_A[ty * dim + tx] = 0;
        }
        /*if ((row < N) && ((i * dim + tx) < N) && ) {
            s_A[ty * dim + tx] = A[row * N + (i * dim + tx)];
        }*/ 

        if (col < N && (i * dim + ty) < N) {
            s_B[ty * dim + tx] = B[(i * dim * N) + (ty * N) + col];
        } 
        else{
            s_B[ty * dim + tx] = 0;
        }

        //debug
        /*printf("Block (%d, %d) Thread (%d, %d): s_A[%d] = %d, s_B[%d] = %d\n",
        blockIdx.y, blockIdx.x,
        threadIdx.y, threadIdx.x,
        ty * dim + tx, s_A[ty * dim + tx],
        ty * dim + tx, s_B[ty * dim + tx]);*/

        __syncthreads();

        // Accumulate the result
        if(row < N && col < N){
            for (int j = 0; j < dim; j++) {
                C[row * N + col] += s_A[ty * dim + j] * s_B[j * dim + tx];
            }
        }
        __syncthreads();
            
    }
        
    //}
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
    printf("Matrix dimension N: %d\n", N);

    size_t size = N * N * sizeof(int);

// Allocate host memory
    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C_cpu = (int *)malloc(size);
    int *C_gpu = (int *)malloc(size);

    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    // print the matrices
    /*printf("Matrix A:\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Matrix B:\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }*/

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
    printf("Threads per block: %d\n", threads);

    // blocks = (N + threads - 1) / threads, instead of N / threads 
    // to ensure all elements are covered if N is not a multiple of threads
    int blocks = (N + threads - 1) / threads;
    printf("Blocks: %d\n", blocks);

    // Set up kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Launch kernel
    mat_mul<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Because kernel launches are asynchronous

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // print the resulting matrix
    /*printf("\nMatrix C:\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%d ", C_gpu[i * N + j]);
        }
        printf("\n");
    }*/

    // Verify the result on the CPU
    verify_mul(A, B, C_cpu, C_gpu, N);

    // Print the first row of A, first col of B and the first entry of the resulting matrices C_cpu and C_gpu
    /*printf("Row 142 of A: ");
    for(int i = 0; i < N; i++){
        printf("%d ", A[142 * N + i]);
    }
    printf("\n");
    printf("Col 0 of B: ");
    for(int i = 0; i < N; i++){
        printf("%d ", B[i * N]);
    }
    printf("\n");*/
    //printf("First element of the resulting matrices (C_cpu, C_gpu): %d, %d\n", C_cpu[0], C_gpu[0]);

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