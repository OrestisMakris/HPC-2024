#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define ARRAY_SIZE 100000     // Large number of iterations for heavy computation
#define BLOCK_SIZE 1000        // Block size for tasks optimized version

// Heavy computation function
double work(int i) {
    double result = 0.0;
    // A heavy computation loop
    for (int j = 0; j < 10000; j++) {
        result += sin(i * j) * cos(j / (i + 1.0)) + sqrt(i + j + 1.0);
    }
    return result;
}

// Function prototypes
void initialize_serial(double *A, int size);
void initialize_threads(double *A, int size);
void initialize_tasks(double *A, int size);
void initialize_tasks_optimized(double *A, int size, int block_size);
double max_difference(double *A, double *B, int size);

// Serial implementation
void initialize_serial(double *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

// OpenMP threads implementation using parallel for
void initialize_threads(double *A, int size) {
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

// OpenMP tasks implementation: one task per iteration
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
            // Wait for all tasks to complete
            #pragma omp taskwait
        }
    }
}

// OpenMP tasks optimized implementation: group iterations into blocks to reduce overhead
void initialize_tasks_optimized(double *A, int size, int block_size) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < size; i += block_size) {
                int end = i + block_size;
                if(end > size)
                    end = size;
                #pragma omp task firstprivate(i, end)
                {
                    for (int j = i; j < end; j++) {
                        A[j] = work(j);
                    }
                }
            }
            #pragma omp taskwait
        }
    }
}

// Compute maximum absolute difference between two arrays
double max_difference(double *A, double *B, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(A[i] - B[i]);
        if(diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

int main() {
    // Allocate separate arrays for each version.
    double *A_serial     = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *A_threads    = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *A_tasks      = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *A_tasks_opt  = (double *)malloc(ARRAY_SIZE * sizeof(double));

    double start, end;
    double time_serial, time_threads, time_tasks, time_tasks_opt;

    // Serial execution
    start = omp_get_wtime();
    initialize_serial(A_serial, ARRAY_SIZE);
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial execution time: %f seconds\n", time_serial);
    printf("Serial execution used 1 core with 1 thread\n\n");

    // Threads execution (parallel for)
    start = omp_get_wtime();
    initialize_threads(A_threads, ARRAY_SIZE);
    end = omp_get_wtime();
    time_threads = end - start;
    printf("Threads (parallel for) execution time: %f seconds\n", time_threads);
    printf("Threads execution used %d cores with up to %d threads\n\n",
           omp_get_num_procs(), omp_get_max_threads());

    // Tasks execution (one task per iteration)
    start = omp_get_wtime();
    initialize_tasks(A_tasks, ARRAY_SIZE);
    end = omp_get_wtime();
    time_tasks = end - start;
    printf("Tasks (1 task per iteration) execution time: %f seconds\n", time_tasks);
    printf("Tasks execution used %d cores with up to %d threads\n\n",
           omp_get_num_procs(), omp_get_max_threads());

    // Optimized tasks execution (chunked tasks)
    start = omp_get_wtime();
    initialize_tasks_optimized(A_tasks_opt, ARRAY_SIZE, BLOCK_SIZE);
    end = omp_get_wtime();
    time_tasks_opt = end - start;
    printf("Optimized Tasks (chunked tasks) execution time: %f seconds\n", time_tasks_opt);
    printf("Optimized Tasks execution used %d cores with up to %d threads\n\n",
           omp_get_num_procs(), omp_get_max_threads());

    // Compare results: calculate the maximum absolute difference between serial and each parallel method.
    double diff_threads   = max_difference(A_serial, A_threads, ARRAY_SIZE);
    double diff_tasks     = max_difference(A_serial, A_tasks, ARRAY_SIZE);
    double diff_tasks_opt = max_difference(A_serial, A_tasks_opt, ARRAY_SIZE);

    printf("Maximum absolute difference compared to serial execution:\n");
    printf("  Threads (parallel for):    %e\n", diff_threads);
    printf("  Tasks (1 task per iteration): %e\n", diff_tasks);
    printf("  Optimized Tasks (chunked): %e\n", diff_tasks_opt);

    free(A_serial);
    free(A_threads);
    free(A_tasks);
    free(A_tasks_opt);

    return 0;
}
