#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "helper_functions.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(int);

    // Initialize matrices A, B, C, D
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C = (int *)malloc(N * N * sizeof(int));
    int *D = (int *)malloc(N * N * sizeof(int));

    // Initialize result matrices E and F
    int *E_cpu = (int *)malloc(N * N * sizeof(int));
    int *E_gpu = (int *)malloc(N * N * sizeof(int));

    int *F_cpu = (int *)malloc(N * N * sizeof(int));
    int *F_gpu = (int *)malloc(N * N * sizeof(int));
    
    // Temporary matrices for optimized multiplications
    int *P = (int *)malloc(size);
    int *Q = (int *)malloc(size);
    int *R = (int *)malloc(size);
    
    // Initialize input matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);
    
    // Start timing using OpenMP
    double start = omp_get_wtime();
    
    // Compute P = (A+B)*C, Q = A*(C-D), R = (A-B)*D concurrently on the GPU:
    // and then compute E = Q + R, F = P - Q
    #pragma omp target data map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) \
                            map(alloc: P[0:N*N], Q[0:N*N], R[0:N*N]) \
                            map(from: E_gpu[0:N*N], F_gpu[0:N*N])
    {
        // Compute P, Q, R on GPU
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int dot_P = 0, dot_Q = 0, dot_R = 0;
                for (int k = 0; k < N; k++) {
                    dot_P += (A[i * N + k] + B[i * N + k]) * C[k * N + j];
                    dot_Q += A[i * N + k] * (C[k * N + j] - D[k * N + j]);
                    dot_R += (A[i * N + k] - B[i * N + k]) * D[k * N + j];
                }
                P[i * N + j] = dot_P;
                Q[i * N + j] = dot_Q;
                R[i * N + j] = dot_R;
            }
        }

        // Compute E and F using P, Q, R on GPU
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i * N + j;
                E_gpu[idx] = Q[idx] + R[idx];
                F_gpu[idx] = P[idx] - Q[idx];
            }
        }
    }
    
    double end = omp_get_wtime();
    printf("GPU Execution Time (optimized): %f seconds\n", end - start);
    
    // Verify the result against CPU computation
    verify_complex_mat_mul(A, B, C, D, E_cpu, E_gpu, F_cpu, F_gpu, N);
    
    // Free allocated memory
    free(A); free(B); free(C); free(D);
    free(E_gpu); free(E_cpu);
    free(F_gpu); free(F_cpu);
    free(P); free(Q); free(R);
    
    return 0;
}
