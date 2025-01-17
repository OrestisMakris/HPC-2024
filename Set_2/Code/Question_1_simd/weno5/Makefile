CC=gcc
CFLAGS=-O3 -Wall -march=native -Wno-unused-variable -fno-tree-vectorize
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench
