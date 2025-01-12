#include <stdlib.h>
#include <omp.h>

void create_random_matrices(int ****matrices, int rank, int NUM_THREADS, int N){
    int thread_id, i, j, k;

    #pragma omp parallel private(thread_id, i, j, k)
    {
        // Allocate memory for this thread's matrix
        thread_id = omp_get_thread_num();
        unsigned int seed = rank * NUM_THREADS + 1 + thread_id; // Unique seed for each process's thread

        matrices[thread_id] = (int ***)malloc(N * sizeof(int **));
        for (i = 0; i < N; i++) {
            matrices[thread_id][i] = (int **)malloc(N * sizeof(int *));
            for (j = 0; j < N; j++) {
                matrices[thread_id][i][j] = (int *)malloc(N * sizeof(int));
            }
        }

        // Initialize the matrix with random values
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    matrices[thread_id][i][j][k] = rand_r(&seed) % 100; // Random number between 0 and 99
                }
            }
        }
    }
}