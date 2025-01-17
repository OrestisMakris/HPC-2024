#include <stdio.h>
#include <omp.h>

#define N 100000

    float A[N], B[N], C[N];


int main() {
    // Declare and allocate arrays

    // Initialize arrays A and B
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Offload the computation to the GPU
//    #pragma omp target map(to: A[0:N], B[0:N]) map(from: C[0:N])
//    #pragma omp target map(to: A[0:N], B[0:N]) map(from: C[0:N]) nowait
//	#pragma omp target teams distribute parallel for simd
//    #pragma omp parallel for simd
    #pragma omp target
    #pragma omp loop
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
    }

    // Check results (printing a few values)
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    return 0;
}
