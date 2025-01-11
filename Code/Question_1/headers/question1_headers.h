#ifndef QUESTION1_HEADERS_H
#define QUESTION1_HEADERS_H

#include <mpi.h>

// Declare the function MPI_Exscan_omp
void MPI_Exscan_omp(int *thread_send, int *thread_total, int *first_thread_recvbuf, MPI_Comm comm);

// Declate the function to validate the matrices.bin file
void validate_file(const char *filename, int ****original_matrices, int* thread_offsets, int N, MPI_Comm comm);

// Declate the function to validate the matrices.bin file
void validate_compressed_file(const char *filename, int ****original_matrices, int* thread_matrix_sizes, int* thread_offsets, int N, MPI_Comm comm);

#endif // MPI_EXSCAN_OMP_H