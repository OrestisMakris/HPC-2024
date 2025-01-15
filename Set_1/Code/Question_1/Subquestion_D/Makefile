CC = mpicc
CFLAGS = -g -fopenmp -Wall -O2 -I../headers
LDFLAGS = -L/usr/lib -lzstd

all: compressed_matrix_io

compressed_matrix_io: compressed_matrix_io.c ../headers/mpi_exscan_omp.o ../headers/validate_compressed.o ../headers/create_random_matrices.o ../headers/print_matrices.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

../headers/mpi_exscan_omp.o: ../headers/mpi_exscan_omp.c ../headers/question1_headers.h ../headers/zstd.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/validate_compressed.o: ../headers/validate_compressed.c ../headers/question1_headers.h ../headers/zstd.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/create_random_matrices.o: ../headers/create_random_matrices.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

../headers/print_matrices.o: ../headers/print_matrices.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f compressed_matrix_io ../headers/mpi_exscan_omp.o ../headers/validate_compressed.o *.bin ../headers/create_random_matrices.o ../headers/print_matrices.o
