CC = mpicc
CFLAGS = -Wall

all: exscan_ptp default_exscan

exscan_ptp: exscan_ptp.c
	$(CC) $(CFLAGS) -o $@ $<

default_exscan: default_exscan.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f exscan_ptp default_exscan

# To run the code, use the following commands:
# mpirun -np <num_processes> ./program