#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>

int main(int argc, char** argv)
{
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided); 
    assert(provided >= MPI_THREAD_FUNNELED);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(2);

    #pragma omp parallel
    {
        #pragma omp critical
        printf("Hello, world. I am process %d of %d and thread=%d\n", rank, size, omp_get_thread_num());
    }

    MPI_Finalize();

    return 0;
}