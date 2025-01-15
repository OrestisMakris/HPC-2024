// Helper functions
#ifndef HELPER_H
#define MHELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to initialize matrix with random integer values
void initialize_matrix(int *matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 10; // Random integers between 0 and 9
    }
}

// Function to initialize matrix with random float values
void initialize_matrix(float *matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = (float)(rand() % 100) / 10.0f; // Random floats between 0.0 and 9.9
    }
}

// Function to print a matrix
void print_matrix(const int* matrix, int N, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to verify a complex matrix multiplication on CPU for integer values
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
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("CPU Execution time: %.3f ms\n", cpu_time);

    // Verify results
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
    printf("Complex Matrix Multiplication is correct!\n");
}

// Function to verify a complex matrix multiplication on CPU for float values
void verify_complex_mat_mul(float* A, float* B, float* C, float* D, float* E_cpu, float* E_gpu, float* F_cpu, float* F_gpu, int N) {
    clock_t start, end;

    // Start timing
    start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum_AC = 0.0f, sum_BD = 0.0f, sum_AD = 0.0f, sum_BC = 0.0f;

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
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("CPU Execution time: %.3f ms\n", cpu_time);

    // Verify results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(E_cpu[i * N + j] - E_gpu[i * N + j]) > 1e-2) {
                printf("Mismatch at index (%d, %d): E_CPU = %.6f, E_GPU = %.6f\n", i, j, E_cpu[i * N + j], E_gpu[i * N + j]);
                return;
            }
            if (fabs(F_cpu[i * N + j] - F_gpu[i * N + j]) > 1e-2) {
                printf("Mismatch at index (%d, %d): F_CPU = %.6f, F_GPU = %.6f\n", i, j, F_cpu[i * N + j], F_gpu[i * N + j]);
                return;
            }
        }
    }
    printf("Complex Matrix Multiplication is correct!\n");
}

#endif 