CC = mpicc
CFLAGS = -fopenmp -Wall -O2

all:  

omp_test: omp_test.c
	$(CC) $(CFLAGS) -o $@ $<

hybrid_test: hybrid_test.c
	$(CC) $(CFLAGS) -o $@ $<

omp_exscan: omp_exscan.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f hybrid_test omp_test omp_exscan

# To run mpi code, use the following commands:
# mpirun -np <num_processes> ./program
