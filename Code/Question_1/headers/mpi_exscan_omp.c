#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include "question1_headers.h"

// Implement the function MPI_Exscan_omp
void MPI_Exscan_omp(int thread_send, int* prev_thread_send, int *thread_totals, int first_thread_recvbuf, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int num_threads = omp_get_num_threads();   
    int thread_id = omp_get_thread_num();

    if (thread_id == 0) {
        if (rank == 0) {
            // The first thread of the first process has no previous process to receive from
            thread_totals[0] = 0;
        } else {
            // The first thread of other processes
            thread_totals[0] = first_thread_recvbuf;
        }
        *prev_thread_send = thread_send + thread_totals[0];
    } 
    else {
        // Update the total for this thread
        thread_totals[thread_id] = *prev_thread_send;
        // Send the total + this threads send value to the next thread
        *prev_thread_send = thread_send + thread_totals[thread_id];
    }

    // The last thread sends this sum to the next process
    if (thread_id == num_threads - 1 && rank != size - 1) {
        MPI_Send(prev_thread_send, 1, MPI_INT, rank + 1, 0, comm);
    }
}