#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int sendbuf = rank + 1;
    int recvbuf = 0;
    
    MPI_Exscan(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    printf("Process %d: sendbuf = %d, recvbuf = %d\n", rank, sendbuf, recvbuf);
    
    MPI_Finalize();
    return 0;
}