#include <mpi.h>
#include <stdio.h>

void Exscan_pt2pt(int *sendbuf, int *recvbuf, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Process 0 receives 0
    if (rank == 0) {
        *recvbuf = 0;
    }
    else {
        MPI_Status status;

        // Get the previous process' sum
        MPI_Recv(recvbuf, 1, MPI_INT, rank-1, 0, comm, &status);    
    }
    
    // Calculate the current process' sum
    int sum = *recvbuf + *sendbuf;
    
    if (rank < size - 1) {
        // Send the sum to the next process
        MPI_Send(&sum, 1, MPI_INT, rank+1, 0, comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int sendbuf = rank + 1;
    int recvbuf;
    
    Exscan_pt2pt(&sendbuf, &recvbuf, MPI_COMM_WORLD);
    
    printf("Process %d: sendbuf = %d, recvbuf = %d\n", rank, sendbuf, recvbuf);
    
    MPI_Finalize();
    return 0;
}

