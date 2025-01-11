# include <stdio.h>
# include <stdlib.h>
# include <omp.h>

#include "question1_headers.h"

// Implement the function to validate the matrices.bin file
void validate_file(const char *filename, int ****original_matrices, int* thread_offsets, int N, MPI_Comm comm){
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    #pragma omp parallel
    {   
        int thread_id = omp_get_thread_num();
        int *read_flat_matrix = (int *)malloc(N * N * N * sizeof(int));

        MPI_File_read_at(file, thread_offsets[thread_id], read_flat_matrix, N * N * N, MPI_INT, MPI_STATUS_IGNORE);

        int idx = 0;
        int valid = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    if (read_flat_matrix[idx++] != original_matrices[thread_id][i][j][k]){
                        valid = 0;
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (valid) {
                printf("Process %d, Thread %d: Validation successful.\n", rank, thread_id);
            } else {
                printf("Process %d, Thread %d: Validation failed.\n", rank, thread_id);
            }
        }

        free(read_flat_matrix);
    }

    MPI_File_close(&file);
}