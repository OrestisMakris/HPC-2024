#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

#define N 1024  // Size of the NxN matrices

// Function to initialize a matrix with random values
void initialize_matrix(double *mat, int size) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = (rand() / (double)RAND_MAX);
    }
}

int main() {
    // Allocate memory for the real and imaginary parts of matrices A, B, C, D
    double *A_real = (double *)malloc(N * N * sizeof(double));
    double *A_imag = (double *)malloc(N * N * sizeof(double));
    double *B_real = (double *)malloc(N * N * sizeof(double));
    double *B_imag = (double *)malloc(N * N * sizeof(double));
    double *C_real = (double *)malloc(N * N * sizeof(double));
    double *C_imag = (double *)malloc(N * N * sizeof(double));
    double *D_real = (double *)malloc(N * N * sizeof(double));
    double *D_imag = (double *)malloc(N * N * sizeof(double));

    // Allocate memory for result matrices E (real) and F (imaginary)
    double *E_real = (double *)malloc(N * N * sizeof(double));
    double *E_imag = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices A, B, C, D with random values
    initialize_matrix(A_real, N);
    initialize_matrix(A_imag, N);
    initialize_matrix(B_real, N);
    initialize_matrix(B_imag, N);
    initialize_matrix(C_real, N);
    initialize_matrix(C_imag, N);
    initialize_matrix(D_real, N);
    initialize_matrix(D_imag, N);

    // Start the clock
    double start_time = omp_get_wtime();

    // Offload computation to the GPU
    #pragma omp target data map(to: A_real[0:N*N], A_imag[0:N*N], \
                                    B_real[0:N*N], B_imag[0:N*N], \
                                    C_real[0:N*N], C_imag[0:N*N], \
                                    D_real[0:N*N], D_imag[0:N*N]) \
                            map(from: E_real[0:N*N], E_imag[0:N*N])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double real_part = 0.0;
                double imag_part = 0.0;

                for (int k = 0; k < N; k++) {
                    // Real part of (A + Bi)(C + Di)
                    real_part += A_real[i * N + k] * C_real[k * N + j] - A_imag[i * N + k] * C_imag[k * N + j] -
                                 B_real[i * N + k] * D_real[k * N + j] + B_imag[i * N + k] * D_imag[k * N + j];

                    // Imaginary part of (A + Bi)(C + Di)
                    imag_part += A_real[i * N + k] * C_imag[k * N + j] + A_imag[i * N + k] * C_real[k * N + j] +
                                 B_real[i * N + k] * D_imag[k * N + j] + B_imag[i * N + k] * D_real[k * N + j];
                }

                E_real[i * N + j] = real_part;
                E_imag[i * N + j] = imag_part;
            }
        }
    }

    // End the clock
    double end_time = omp_get_wtime();
    printf("Complex matrix multiplication completed in %f seconds.\n", end_time - start_time);

    // Free allocated memory
    free(A_real);
    free(A_imag);
    free(B_real);
    free(B_imag);
    free(C_real);
    free(C_imag);
    free(D_real);
    free(D_imag);
    free(E_real);
    free(E_imag);

    return 0;
}
