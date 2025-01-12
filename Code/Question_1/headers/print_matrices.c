#include <stdio.h>

void print_matrices(int**** matrices, int rank, int NUM_THREADS, int N){
    // Print matrices for verification
    for (int t = 0; t < NUM_THREADS; t++) {
        printf("Process %d:\n", rank);
        printf("Matrix for thread %d:\n", t);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    printf("%d ", matrices[t][i][j][k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}