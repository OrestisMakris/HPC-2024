#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int send = rank + 1;
    int total = 0;
    
    MPI_Exscan(&send, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    printf("Process %d: Sent = %d, Partial Reduction = %d\n", rank, send, total);
    
    MPI_Finalize();
    return 0;
}