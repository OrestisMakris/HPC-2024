#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

# define NUM_THREADS 4

void MPI_Exscan_omp(int *thread_send, int *thread_recv, int *recv_buf, int rank){
    #pragma omp parallel 
    {
        int thread_num = omp_get_thread_num();

        // Calculate the starting value for this MPI process based on its rank.
        // Each thread within the process computes its send value by adding its thread ID to the starting value.
        int start = rank * omp_get_num_threads() + 1;
        thread_send[thread_num] = start + thread_num;

        // Barrier to ensure all threads have set their values
        #pragma omp barrier

        // This block is essentially sequential
        #pragma omp for ordered
        for(int thread_num = 0; thread_num < omp_get_num_threads(); thread_num++){
            #pragma omp ordered
            {
                // Thread 0 should receive the previous processes' sum from *recvbuf
                if(thread_num == 0){
                    thread_recv[0] = *recv_buf;
                }
                // The rest of the threads should receive 
                // the sum of the previous thread's send value and the previous thread's receive value
                else{
                    thread_recv[thread_num] = thread_send[thread_num - 1] + thread_recv[thread_num - 1];
                }
            }
        }
        // A single thread prints the results
        #pragma omp single
        {
            for(int i = 0; i < omp_get_num_threads(); i++){
                printf("Process: %d Thread %d: Sent: %d and Received: %d\n", rank, i, thread_send[i], thread_recv[i]);
            }
        }
        
    }
}

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
    int *thread_recv = (int *) malloc(omp_get_num_threads() * sizeof(int));

    // Receive buffer for the first thread of the MPI process
    int recvbuf = 0;

    if(rank==0){
        recvbuf = 0;
    }
    // Receive the sum from the last thread of the previous process
    else{
        MPI_Status status;
        MPI_Recv(&recvbuf, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Exscan_omp(thread_send, thread_recv, &recvbuf, rank);

    // Send the last thread's sum to the next process
    if(rank < size - 1){
        int sum = thread_recv[NUM_THREADS-1] + thread_send[NUM_THREADS-1];

        MPI_Send(&sum, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }

    free(thread_send);
    free(thread_recv);

    MPI_Finalize();

    return 0;
}