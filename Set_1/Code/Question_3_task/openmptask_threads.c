#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define ARRAY_SIZE 1000000  // Large number of iterations for heavy computation
#define BLOCK_SIZE 1000     // Grain size or block size for tasks

//-------------------------
// Heavy computation function
//-------------------------
double work(int i) {
    double result = 0.0;
    for (int j = 0; j < 10000; j++) { // Heavy inner loop
        result += sin(i * j) * cos(j / (i + 1.0)) + sqrt(i + j + 1.0);
    }
    return result;
}

//-------------------------
// Function Prototypes
//-------------------------
void initialize_serial(double *A, int size);
void initialize_tasks_optimized(double *A, int size, int block_size);
void initialize_tasks_taskloop(double *A, int size, int block_size);
double compute_sum_tasks(int size, int block_size);
double max_difference(double *A, double *B, int size);

//-------------------------
// Serial implementation
//-------------------------
void initialize_serial(double *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

//-------------------------
// OpenMP Optimized Tasks:
// Group iterations into blocks to reduce the overhead of creating many tasks.
// Also, we add the 'untied' clause to allow flexibility in task scheduling.
//-------------------------
void initialize_tasks_optimized(double *A, int size, int block_size) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int i = 0; i < size; i += block_size) {
                int end = i + block_size;
                if (end > size) end = size;
                #pragma omp task firstprivate(i, end) untied
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

//-------------------------
// OpenMP Taskloop Implementation:
// Uses the 'taskloop' directive to automatically break the loop into tasks.
// The 'grainsize' clause specifies the minimum number of iterations per task,
// and the 'untied' clause gives more scheduling flexibility.
//-------------------------
void initialize_tasks_taskloop(double *A, int size, int block_size) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp taskloop grainsize(block_size) untied
            for (int i = 0; i < size; i++) {
                A[i] = work(i);
            }
        }
    }
}

//-------------------------
// OpenMP Tasks with Reduction:
// Computes a global sum over all work(i) values using a taskloop with reduction.
// Note: Task reduction is available in OpenMP 5.0 and later.
//-------------------------
double compute_sum_tasks(int size, int block_size) {
    double sum = 0.0;
    #pragma omp parallel reduction(+:sum)
    {
        #pragma omp single nowait
        {
            #pragma omp taskloop grainsize(block_size) untied reduction(+:sum)
            for (int i = 0; i < size; i++) {
                sum += work(i);
            }
        }
    }
    return sum;
}

//-------------------------
// Utility: Compute maximum absolute difference between two arrays
//-------------------------
double max_difference(double *A, double *B, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(A[i] - B[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

//-------------------------
// Main function: Compare various implementations and their timings/results
//-------------------------
int main() {
    // Allocate arrays for serial, taskloop, and optimized tasks implementations.
    double *A_serial    = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *A_taskloop  = (double *)malloc(ARRAY_SIZE * sizeof(double));
    double *A_tasks_opt = (double *)malloc(ARRAY_SIZE * sizeof(double));

    double start, end;
    double time_serial, time_taskloop, time_tasks_opt, time_sum;
    double sum_tasks;

    //--- Serial Execution ---
    start = omp_get_wtime();
    initialize_serial(A_serial, ARRAY_SIZE);
    end = omp_get_wtime();
    time_serial = end - start;
    printf("Serial execution time: %f seconds\n", time_serial);

    //--- OpenMP Taskloop Execution ---
    start = omp_get_wtime();
    initialize_tasks_taskloop(A_taskloop, ARRAY_SIZE, BLOCK_SIZE);
    end = omp_get_wtime();
    time_taskloop = end - start;
    printf("Taskloop execution time: %f seconds\n", time_taskloop);
    double diff_taskloop = max_difference(A_serial, A_taskloop, ARRAY_SIZE);
    printf("Taskloop max difference vs serial: %e\n", diff_taskloop);

    //--- OpenMP Optimized Tasks Execution (Block-based Tasks) ---
    start = omp_get_wtime();
    initialize_tasks_optimized(A_tasks_opt, ARRAY_SIZE, BLOCK_SIZE);
    end = omp_get_wtime();
    time_tasks_opt = end - start;
    printf("Optimized tasks execution time: %f seconds\n", time_tasks_opt);
    double diff_tasks_opt = max_difference(A_serial, A_tasks_opt, ARRAY_SIZE);
    printf("Optimized tasks max difference vs serial: %e\n", diff_tasks_opt);

    //--- OpenMP Tasks with Reduction for Global Sum ---
    start = omp_get_wtime();
    sum_tasks = compute_sum_tasks(ARRAY_SIZE, BLOCK_SIZE);
    end = omp_get_wtime();
    time_sum = end - start;
    printf("Tasks reduction (global sum) execution time: %f seconds, sum = %f\n", time_sum, sum_tasks);

    // Cleanup
    free(A_serial);
    free(A_taskloop);
    free(A_tasks_opt);

    return 0;
}
