# Compiler
CC = gcc

# Compiler Flags
CFLAGS = -fopenmp -foffload=nvptx-none -fcf-protection=none -no-pie

# Build Targets
all: mat_add mat_mul

mat_add: mat_add.c
	$(CC) $(CFLAGS) -o $@ $^

mat_mul: mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^

# Clean Targets
clean:
	rm -f mat_add mat_mul
