# Compiler
CC = gcc

# Compiler Flags
CFLAGS = -fopenmp -foffload=nvptx-none -fcf-protection=none -no-pie

# Build Targets
all:complex_mat_mul

complex_mat_mul: complex_mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^

# Clean Targets
clean:
	rm -f complex_mat_mul
