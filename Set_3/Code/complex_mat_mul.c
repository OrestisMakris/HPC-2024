#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "helper_functions.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }

    // Set square matrix dimension (NxN)
    int N = atoi(argv[1]);

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

    // Start timing the GPU execution
    double start = omp_get_wtime();

    #pragma omp target data map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) map(from: E_gpu[0:N*N], F_gpu[0:N*N])
    {
        //#pragma omp target parallel for
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Initialze temp dot product variables at index (i,j) for each matrix multiplication
                int dot_AC = 0, dot_BD = 0;

                int dot_AD = 0, dot_BC = 0;

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
