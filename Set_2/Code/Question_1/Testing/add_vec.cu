# include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

# define N 512
# define THREADS_PER_BLOCK 512

void random_ints(int *a, int n){
    for(int i = 0; i < n; i++){
        a[i] = rand() % 100;
    }
}

__global__ void add(int *a, int *b, int *c){
int index = blockIdx.x * blockDim.x + threadIdx.x;
c[index] = a[index] + b[index];
}

int main(void){
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    // cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    printf("First 10 elements of a are: ");
    for(int i = 0; i < 10; i++){
        printf("%d ", a[i]);
    }
    printf("\n");
    printf("First 10 elements of b are: ");
    for(int i = 0; i < 10; i++){
        printf("%d ", b[i]);
    }
    printf("\n");

    // Copy inputs to device
    // cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    printf("First 10 elements of the result are: ");
    for(int i = 0; i < 10; i++){
        printf("%d ", c[i]);
    }

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}