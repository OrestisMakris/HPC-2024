#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <zstd.h>

void decompress_data(const void* src, size_t src_size, void* dest, size_t dest_size) {
    size_t decompressed_size = ZSTD_decompress(dest, dest_size, src, src_size);
    if (ZSTD_isError(decompressed_size)) {
        fprintf(stderr, "Decompression failed: %s\n", ZSTD_getErrorName(decompressed_size));
        exit(EXIT_FAILURE);
    }
    if (decompressed_size != dest_size) {
        fprintf(stderr, "Decompressed size mismatch\n");
        exit(EXIT_FAILURE);
    }
}

void validate_compressed_file(const char *filename, int ****original_matrices, int* thread_matrix_sizes, int* thread_offsets, int N, MPI_Comm comm){
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    if(rank == 0){
        printf("\nValidating compressed file.\n");
    }
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *decompressed_matrix = (int *)malloc(N * N * N * sizeof(int));
        void* compressed_data = malloc(thread_matrix_sizes[thread_id]);

        MPI_File_read_at(file, thread_offsets[thread_id], compressed_data, thread_matrix_sizes[thread_id], MPI_BYTE, MPI_STATUS_IGNORE);
        decompress_data(compressed_data, thread_matrix_sizes[thread_id], decompressed_matrix, N * N * N * sizeof(int));

        // Validate decompressed matrix
        int idx = 0;
        int valid = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    if (decompressed_matrix[idx++] != original_matrices[thread_id][i][j][k]) {
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

        free(decompressed_matrix);
        free(compressed_data);
    }
}