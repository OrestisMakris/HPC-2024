
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "helper_functions.h"

/*void compute_E(float* A, float* B, float* C, float* D, float* E, int N) {
    #pragma omp target parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot_AC = 0.0f, dot_BD = 0.0f;

            // Compute the dot products for E at index (i, j)
            for (int k = 0; k < N; k++) {
                dot_AC += A[i * N + k] * C[k * N + j];
                dot_BD += B[i * N + k] * D[k * N + j];
            }

            // Compute E
            E[i * N + j] = dot_AC - dot_BD;
        }
    }
}

void compute_F(float* A, float* B, float* C, float* D, float* F, int N) {
    #pragma omp target parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot_AD = 0.0f, dot_BC = 0.0f;

            // Compute the dot products for F at index (i, j)
            for (int k = 0; k < N; k++) {
                dot_AD += A[i * N + k] * D[k * N + j];
                dot_BC += B[i * N + k] * C[k * N + j];
            }

            // Compute F
            F[i * N + j] = dot_AD + dot_BC;
        }
    }
}*/

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }

    // Set square matrix dimension (NxN)
    int N = atoi(argv[1]);

    // Initialize matrices A, B, C, D
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    float *D = (float *)malloc(N * N * sizeof(float));

    // Initialize result matrices E and F
    float *E_cpu = (float *)malloc(N * N * sizeof(float));
    float *E_gpu = (float *)malloc(N * N * sizeof(float));

    float *F_cpu = (float *)malloc(N * N * sizeof(float));
    float *F_gpu = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices A , C, D
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);

    // Print Matrices
    /*print_matrix(A, N, "Matrix A");
    print_matrix(B, N, "Matrix B");
    print_matrix(C, N, "Matrix C");
    print_matrix(D, N, "Matrix D");*/

	// Compute E = AC - BD and F = AD + BC
    omp_set_default_device(0);
    printf("Using OpenMP Offloading: %s\n", omp_is_initial_device() ? "NO (CPU)" : "YES (GPU)");


    // Start timing the GPU execution
    double start = omp_get_wtime();

    #pragma omp target data map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) map(from: E_gpu[0:N*N], F_gpu[0:N*N])
    {
        //#pragma omp target parallel for
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Initialze temp dot product variables at index (i,j) for each matrix multiplication
                float dot_AC = 0.0f, dot_BD = 0.0f;

                float dot_AD = 0.0f, dot_BC = 0.0f;

                // Compute the dot products for E and F at index (i, j)
                for (int k = 0; k < N; k++) {
                    dot_AC += A[i * N + k] * C[k * N + j];
                    dot_BD += B[i * N + k] * D[k * N + j];

                    dot_AD += A[i * N + k] * D[k * N + j];
                    dot_BC += B[i * N + k] * C[k * N + j];
                }
                // Compute E and F at index (i,j)
                E_gpu[i * N + j] = dot_AC - dot_BD;
                F_gpu[i * N + j] = dot_AD + dot_BC;
            }
        }
    }

    // End timing
    double end = omp_get_wtime();
    printf("GPU Execution Time: %f seconds\n", end - start);

    //print_matrix(E_gpu, N, "Matrix E (GPU)");
    //print_matrix(F_gpu, N, "Matrix F (GPU)");

    verify_complex_mat_mul(A, B, C, D, E_cpu, E_gpu, F_cpu, F_gpu, N);

    //print_matrix(E_cpu, N, "Matrix E (CPU)");
    //print_matrix(F_cpu, N, "Matrix F (CPU)");

    // Free memory
    free(A); free(B); free(C); free(D);
    free(E_cpu); free(E_gpu);
    free(F_cpu); free(F_gpu);

    return 0;
}
