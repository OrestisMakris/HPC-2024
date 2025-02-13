# Compiler
CC = gcc

# Compiler Flags
# CFLAGS = -fopenmp -foffload=disable -fcf-protection=none
# CFLAGS = -fopenmp -foffload=nvptx-none -march=native -fcf-protection=none
# CFLAGS = -fopenmp -foffload=nvptx-none -fcf-protection=none 
CFLAGS = -fopenmp -lgomp -foffload=nvptx-none -lcuda -foffload="-misa=sm_35" -fcf-protection=none -fno-stack-protector

# Build Targets
all:complex_mat_mul strength_reduct_compl_mat_mul tiled_complex_mat_mul

complex_mat_mul: complex_mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

strength_reduct_compl_mat_mul: strength_reduct_compl_mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tiled_complex_mat_mul: tiled_complex_mat_mul.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Clean Targets
clean:
	rm -f complex_mat_mul strength_reduct_compl_mat_mul tiled_complex_mat_mul
