#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include "question1_headers.h"

// Implement the function MPI_Exscan_omp
void MPI_Exscan_omp(int *thread_send, int *thread_total, int *first_thread_recvbuf, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    #pragma omp parallel
    {
        // Sequential reduction using `#pragma omp ordered`
        #pragma omp for ordered
        for (int thread_num = 0; thread_num < omp_get_num_threads(); thread_num++) {
            #pragma omp ordered
            {
                if (thread_num == 0) {
                    thread_total[0] = *first_thread_recvbuf;
                } else {
                    thread_total[thread_num] = thread_send[thread_num - 1] + thread_total[thread_num - 1];
                }
            }
        }

        // A single thread prints the results and sends the last thread's sum to the next process
        #pragma omp single
        {
            // Print the results
            for (int i = 0; i < omp_get_num_threads(); i++) {
                printf("Process: %d Thread %d: Sent: %d, Partial Reduction: %d\n",
                       rank, i, thread_send[i], thread_total[i]);
            }

            // Send the last thread's sum to the next process
            if(rank < size - 1){
            int num_threads = omp_get_num_threads();
            int sum = thread_total[num_threads-1] + thread_send[num_threads-1];

            MPI_Send(&sum, 1, MPI_INT, rank+1, 0, comm);
            }
        }
    }
}