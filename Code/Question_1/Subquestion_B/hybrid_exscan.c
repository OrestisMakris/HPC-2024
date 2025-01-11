#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

#include "../headers/question1_headers.h"

# define NUM_THREADS 4

/*
Inter thread communication in an MPI process 
is done using global arrays thread_send and thread_sum, that all threads can access. 
*/

int main(int argc, char** argv)
{
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided); 
    assert(provided >= MPI_THREAD_FUNNELED);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(NUM_THREADS);

    // Initialize the global sum array with the size of the number of threads
    int thread_sums[NUM_THREADS];

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

   printf("\nExscan results for process %d:\n", rank);
   int prev_thread_send = 0;
   #pragma omp parallel
   {
        #pragma omp for ordered
        for(int i = 0; i < omp_get_num_threads(); i++){
            # pragma omp ordered
            {
                int thread_id = omp_get_thread_num();
                int thread_send = rank * omp_get_num_threads() + 1 + thread_id;
                MPI_Exscan_omp(thread_send, &prev_thread_send, thread_sums, first_thread_recvbuf, MPI_COMM_WORLD);
                printf("Process: %d Thread %d: Sent: %d, Partial Reduction: %d\n",rank, i, thread_send, thread_sums[i]);
            }
        }
   }
    MPI_Finalize();

    return 0;
}