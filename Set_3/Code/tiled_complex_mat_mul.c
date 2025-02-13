#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "helper_functions.h"

#define TILE_SIZE 16  

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(int);

    // Allocate matrices A, B, C, D
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C = (int *)malloc(N * N * sizeof(int));
    int *D = (int *)malloc(N * N * sizeof(int));

    // Allocate result matrices E and F
    int *E_cpu = (int *)malloc(N * N * sizeof(int));
    int *E_gpu = (int *)malloc(N * N * sizeof(int));

    int *F_cpu = (int *)malloc(N * N * sizeof(int));
    int *F_gpu = (int *)malloc(N * N * sizeof(int));

    initialize_matrix(A, N);
    initialize_matrix(B, N);
    initialize_matrix(C, N);
    initialize_matrix(D, N);

    double start = omp_get_wtime();

    #pragma omp target data map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) \
                            map(from: E_gpu[0:N*N], F_gpu[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                // Temporary tile buffers
                int local_E[TILE_SIZE][TILE_SIZE] = {{0}};
                int local_F[TILE_SIZE][TILE_SIZE] = {{0}};
                int tileA[TILE_SIZE][TILE_SIZE], tileB[TILE_SIZE][TILE_SIZE];
                int tileC[TILE_SIZE][TILE_SIZE], tileD[TILE_SIZE][TILE_SIZE];

                for (int k = 0; k < N; k += TILE_SIZE) {
                    // Load tile for A and B
                    for (int ti = 0; ti < TILE_SIZE; ti++) {
                        for (int tk = 0; tk < TILE_SIZE; tk++) {
                            int global_row = i + ti, global_col = k + tk;
                            if (global_row < N && global_col < N) {
                                tileA[ti][tk] = A[global_row * N + global_col];
                                tileB[ti][tk] = B[global_row * N + global_col];
                            } else {
                                tileA[ti][tk] = 0;
                                tileB[ti][tk] = 0;
                            }
                        }
                    }
                    // Load tile for C and D
                    for (int tk = 0; tk < TILE_SIZE; tk++) {
                        for (int tj = 0; tj < TILE_SIZE; tj++) {
                            int global_row = k + tk, global_col = j + tj;
                            if (global_row < N && global_col < N) {
                                tileC[tk][tj] = C[global_row * N + global_col];
                                tileD[tk][tj] = D[global_row * N + global_col];
                            } else {
                                tileC[tk][tj] = 0;
                                tileD[tk][tj] = 0;
                            }
                        }
                    }
                    // Compute partial tile results
                    for (int ti = 0; ti < TILE_SIZE; ti++) {
                        for (int tj = 0; tj < TILE_SIZE; tj++) {
                            int sumAC = 0, sumBD = 0;
                            int sumAD = 0, sumBC = 0;
                            for (int tk = 0; tk < TILE_SIZE; tk++) {
                                sumAC += tileA[ti][tk] * tileC[tk][tj];
                                sumBD += tileB[ti][tk] * tileD[tk][tj];
                                sumAD += tileA[ti][tk] * tileD[tk][tj];
                                sumBC += tileB[ti][tk] * tileC[tk][tj];
                            }
                            local_E[ti][tj] += sumAC - sumBD;
                            local_F[ti][tj] += sumAD + sumBC;
                        }
                    }
                }
                // Write back computed tile to global memory
                for (int ti = 0; ti < TILE_SIZE; ti++) {
                    for (int tj = 0; tj < TILE_SIZE; tj++) {
                        int global_row = i + ti, global_col = j + tj;
                        if (global_row < N && global_col < N) {
                            E_gpu[global_row * N + global_col] = local_E[ti][tj];
                            F_gpu[global_row * N + global_col] = local_F[ti][tj];
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    printf("GPU Execution Time (Tiled): %f seconds\n", end - start);

    verify_complex_mat_mul(A, B, C, D, E_cpu, E_gpu, F_cpu, F_gpu, N);

    // Cleanup and verification omitted for brevity
    free(A); free(B); free(C); free(D);
    free(E_cpu); free(E_gpu);
    free(F_cpu); free(F_gpu);

    return 0;
}