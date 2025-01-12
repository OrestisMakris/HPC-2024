CC = mpicc
CFLAGS = -g -fopenmp -Wall -O2 -I../headers

all: hybrid_matrix_io

hybrid_matrix_io: hybrid_matrix_io.c ../headers/mpi_exscan_omp.o ../headers/validate_file.o ../headers/create_random_matrices.o ../headers/print_matrices.o 
	$(CC) $(CFLAGS) -o $@ $^

../headers/mpi_exscan_omp.o: ../headers/mpi_exscan_omp.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/validate_file.o: ../headers/validate_file.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/create_random_matrices.o: ../headers/create_random_matrices.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/print_matrices.o: ../headers/print_matrices.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f hybrid_matrix_io ../headers/mpi_exscan_omp.o ../headers/validate_file.o ../headers/create_random_matrices.o ../headers/print_matrices.o *.bin


# To run mpi code, use the following commands:
# mpirun -np <num_processes> ./program