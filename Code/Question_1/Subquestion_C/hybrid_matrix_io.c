#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

#include "../headers/question1_headers.h"

# define N 3 // Matrix size is N x N x N

int main(int argc, char** argv)
{
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

    create_random_matrices(matrices, rank, NUM_THREADS, N);

    // Print matrices for verification
    print_matrices(matrices, rank, NUM_THREADS, N);

    // Prepare MPI I/O
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "matrices.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Compute the offsets using MPI_Exscan_omp
    //int thread_send[NUM_THREADS];
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
                    int thread_send = N * N * N * sizeof(int);
                    MPI_Exscan_omp(thread_send, &prev_thread_send, thread_offsets, first_thread_recvbuf, MPI_COMM_WORLD);
                    printf("Process: %d Thread %d: Sent: %d, Partial Reduction: %d\n",rank, i, thread_send, thread_offsets[i]);
                }
            }
    }

    // Write each thread's matrix to the binary file
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

        MPI_Offset offset = thread_offsets[thread_id];
        MPI_File_write_at(file, offset, flat_matrix, N * N * N, MPI_INT, MPI_STATUS_IGNORE);

        free(flat_matrix);
    }

    MPI_File_close(&file);

    // Validate the file
    validate_file("matrices.bin", matrices, thread_offsets, N, MPI_COMM_WORLD);

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