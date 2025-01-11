#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

#include "../headers/question1_headers.h"

# define NUM_THREADS 4

int main(int argc, char** argv)
{
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided); 
    assert(provided >= MPI_THREAD_FUNNELED);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(NUM_THREADS);

    // Initialize the send array of ints with the size of the number of threads
    int *thread_send = (int *) malloc(omp_get_num_threads() * sizeof(int));

    // Initialize the receive array of ints with the size of the number of threads
    int *thread_sum = (int *) malloc(omp_get_num_threads() * sizeof(int));

    // Receive buffer for the first thread of the MPI process
    int first_thread_recvbuf;

    if(rank==0){
        // The first thread of the first process has no previous process to receive from
        first_thread_recvbuf = 0;
    }
    // Receive the sum from the last thread of the previous process
    else{
        MPI_Status status;
        MPI_Recv(&first_thread_recvbuf, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
    }

    // Calculate some values for each thread to send
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        // Calculate the starting value for this MPI process based on its rank.
        int start = rank * omp_get_num_threads() + 1;
        thread_send[thread_id] = start + thread_id;
    }

    MPI_Exscan_omp(thread_send, thread_sum, &first_thread_recvbuf, MPI_COMM_WORLD);

    free(thread_send);
    free(thread_sum);

    MPI_Finalize();

    return 0;
}