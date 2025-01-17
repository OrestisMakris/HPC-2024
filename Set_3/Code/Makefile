# Compiler
CC = gcc

# Compiler Flags
# CFLAGS = -fopenmp -foffload=disable -fcf-protection=none
CFLAGS = -fopenmp -foffload=nvptx-none -march=native -fcf-protection=none
# CFLAGS = -fopenmp -foffload=nvptx-none -fcf-protection=none 

# Build Targets
all:complex_mat_mul 

complex_mat_mul: complex_mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Clean Targets
clean:
	rm -f complex_mat_mul 
