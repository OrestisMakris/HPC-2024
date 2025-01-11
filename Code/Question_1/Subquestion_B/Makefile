CC = mpicc
CFLAGS = -fopenmp -Wall -O2 -I../headers

all: omp_test hybrid_test hybrid_exscan

omp_test: omp_test.c
	$(CC) $(CFLAGS) -o $@ $<

hybrid_test: hybrid_test.c
	$(CC) $(CFLAGS) -o $@ $<

hybrid_exscan: hybrid_exscan.c ../headers/mpi_exscan_omp.o
	$(CC) $(CFLAGS) -o $@ $^

../headers/mpi_exscan_omp.o: ../headers/mpi_exscan_omp.c ../headers/question1_headers.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f hybrid_test omp_test hybrid_exscan ../headers/mpi_exscan_omp.o

# To run mpi code, use the following commands:
# mpirun -np <num_processes> ./program
