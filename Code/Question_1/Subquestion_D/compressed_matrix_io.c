#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <zstd.h>

#include "../headers/question1_headers.h"

#define N 3 // Matrix size is N x N x N

size_t compress_data(const void* src, size_t src_size, void** dest) {
    size_t dest_size = ZSTD_compressBound(src_size); // Maximum possible size for compressed data
    *dest = malloc(dest_size);
    if (*dest == NULL) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    size_t compressed_size = ZSTD_compress(*dest, dest_size, src, src_size, 1);
    if (ZSTD_isError(compressed_size)) {
        fprintf(stderr, "Compression failed: %s\n", ZSTD_getErrorName(compressed_size));
        free(*dest);
        exit(EXIT_FAILURE);
    }

    return compressed_size;
}

int main(int argc, char** argv) {
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert(provided >= MPI_THREAD_FUNNELED);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check and parse command-line arguments for NUM_THREADS
    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int NUM_THREADS = atoi(argv[1]);
    if (NUM_THREADS <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of threads must be a positive integer.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    omp_set_num_threads(NUM_THREADS);

    // Allocate memory for the matrices array
    int ****matrices = (int ****)malloc(NUM_THREADS * sizeof(int ***));

    // Allocate memory for the compressed matrices
    void* compressed_matrices[NUM_THREADS];

    create_random_matrices(matrices, rank, NUM_THREADS, N);

    // Print matrices for verification
    print_matrices(matrices, rank, NUM_THREADS, N);

    int thread_matrix_sizes[NUM_THREADS];
    int thread_offsets[NUM_THREADS];

    // Receive buffer for the first thread of the MPI process
    int first_thread_recvbuf;

    if(rank==0){
        // The first thread of the first process has no previous process to receive from
        first_thread_recvbuf = 0;
    }
    // Receive the offset from the last thread of the previous process
    else{
        MPI_Status status;
        MPI_Recv(&first_thread_recvbuf, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
    }

    // Compress each thread's matrix and calculate the size of the compressed data
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *flat_matrix = (int *)malloc(N * N * N * sizeof(int));
        int idx = 0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    flat_matrix[idx++] = matrices[thread_id][i][j][k];
                }
            }
        }

        // Compress the matrix and get the size of the compressed data
        thread_matrix_sizes[thread_id] = (int)compress_data(flat_matrix, N * N * N * sizeof(int), &compressed_matrices[thread_id]);

        free(flat_matrix);
    }

    // Get the offset for each thread
    if(rank == 0){
        printf("\nExscan results:\n");
    }
    int prev_thread_send = 0;
    #pragma omp parallel
    {
            #pragma omp for ordered
            for(int i = 0; i < omp_get_num_threads(); i++){
                # pragma omp ordered
                {
                    int thread_send = thread_matrix_sizes[i];
                    MPI_Exscan_omp(thread_send, &prev_thread_send, thread_offsets, first_thread_recvbuf, MPI_COMM_WORLD);
                    printf("Process: %d Thread %d: Sent: %d, Partial Reduction: %d\n",rank, i, thread_send, thread_offsets[i]);
                }
            }
    }

    // Get the offset for each thread
    //MPI_Exscan_omp(thread_matrix_sizes, thread_offsets, &first_thread_recvbuf, MPI_COMM_WORLD);

    // Prepare MPI I/O
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "matrices_compressed.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    
    // Write each thread's compressed matrix to the binary file
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int offset = thread_offsets[thread_id];
        MPI_File_write_at(file, offset, compressed_matrices[thread_id], thread_matrix_sizes[thread_id], MPI_BYTE, MPI_STATUS_IGNORE);

        free(compressed_matrices[thread_id]);
    }

    MPI_File_close(&file);

    // Decompress and validate the matrices
    validate_compressed_file("matrices_compressed.bin", matrices, thread_matrix_sizes, thread_offsets, N, MPI_COMM_WORLD);

    MPI_File_close(&file);

    // Free allocated memory
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                free(matrices[t][i][j]);
            }
            free(matrices[t][i]);
        }
        free(matrices[t]);
    }
    free(matrices);

    MPI_Finalize();

    return 0;
}
