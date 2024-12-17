#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

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
    int *send = (int *) malloc(omp_get_num_threads() * sizeof(int));

    // Initialize the receive array of ints with the size of the number of threads
    int *recv = (int *) malloc(omp_get_num_threads() * sizeof(int));

    #pragma omp parallel 
    {
        int thread_num = omp_get_thread_num();

        // Set the value of the send array at the thread number index to the thread number + 1
        // Eg. Thread 0 will send 1, Thread 1 will send 2, etc.
        send[thread_num] = thread_num + 1;

        // Barrier to ensure all threads have set their values
        #pragma omp barrier

        #pragma omp for ordered
        for(int thread_num = 0; thread_num < omp_get_num_threads(); thread_num++){
            #pragma omp ordered
            {
                // Thread 0 should be receive 0
                if(thread_num == 0){
                    recv[0] = 0;
                }
                // The rest of the threads should receive 
                // the sum of the previous thread's send value and the previous thread's receive value
                else{
                    recv[thread_num] = send[thread_num - 1] + recv[thread_num - 1];
                }
            }
        }
        // A single thread prints the results
        #pragma omp single
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
            for(int i = 0; i < omp_get_num_threads(); i++){
                printf("Thread %d: Sent: %d and Received: %d\n", i, send[i], recv[i]);
            }
        }
        
    }

    free(send);
    free(recv);

    MPI_Finalize();

    return 0;
}