#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define ARRAY_SIZE 100 // Large number of iterations for heavy computation

// Heavy computation function
double work(int i) {
    double result = 0.0;
    for (int j = 0; j < 10000; j++) { // Increased iterations
        result += sin(i * j) * cos(j / (i + 1.0)) + sqrt(i + j + 1.0);
    }
    return result;
}

// Function prototypes
void initialize_serial(double *A, int size);
void initialize_threads(double *A, int size);
void initialize_tasks(double *A, int size);

// Serial implementation
void initialize_serial(double *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

// OpenMP threads implementation
void initialize_threads(double *A, int size) {
#pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

// OpenMP tasks implementation
void initialize_tasks(double *A, int size) {
#pragma omp parallel
    {
#pragma omp single
        {
            for (int i = 0; i < size; i++) {
#pragma omp task firstprivate(i)
                {
                    A[i] = work(i);
                }
            }
        }
    }
}

int main() {
    double *A = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double start, end;

    // Serial execution
    start = omp_get_wtime();
    initialize_serial(A, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Serial execution time: %f seconds\n", end - start);

    // Threads execution
    start = omp_get_wtime();
    initialize_threads(A, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Threads execution time: %f seconds\n", end - start);

    // Tasks execution
    start = omp_get_wtime();
    initialize_tasks(A, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Tasks execution time: %f seconds\n", end - start);

    free(A);
    return 0;
}
