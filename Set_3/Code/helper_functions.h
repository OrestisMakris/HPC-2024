// Helper functions
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function to initialize matrix with random int values
void initialize_matrix(int* matrix, int N) {
    for(int i = 0; i < N * N; i++){
         matrix[i] = rand() % 100; // Random integers between 0 and 99
    }
}

// Function to print a matrix
void print_matrix(const int* matrix, int N, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d\t", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to verify a complex matrix multiplication on CPU for int values
void verify_complex_mat_mul(int* A, int* B, int* C, int* D, int* E_cpu, int* E_gpu, int* F_cpu, int* F_gpu, int N) {
    clock_t start, end;

    // Start timing
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum_AC = 0, sum_BD = 0, sum_AD = 0, sum_BC = 0;

            // Compute the dot products for E and F at index (i, j)
            for (int k = 0; k < N; k++) {
                sum_AC += A[i * N + k] * C[k * N + j];
                sum_BD += B[i * N + k] * D[k * N + j];

                sum_AD += A[i * N + k] * D[k * N + j];
                sum_BC += B[i * N + k] * C[k * N + j];
            }

            // Compute E and F
            E_cpu[i * N + j] = sum_AC - sum_BD;
            F_cpu[i * N + j] = sum_AD + sum_BC;
        }
    }

    // End timing
    end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Execution time: %f seconds\n", cpu_time);

    // Verify results
    printf("Verifying results...\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (E_cpu[i * N + j] != E_gpu[i * N + j]) {
                printf("Mismatch at index (%d, %d): E_CPU = %d, E_GPU = %d\n", i, j, E_cpu[i * N + j], E_gpu[i * N + j]);
                return;
            }
            if (F_cpu[i * N + j] != F_gpu[i * N + j]) {
                printf("Mismatch at index (%d, %d): F_CPU = %d, F_GPU = %d\n", i, j, F_cpu[i * N + j], F_gpu[i * N + j]);
                return;
            }
        }
    }
    printf("Complex Matrix Multiplication is correct!\n\n");
}

#endif