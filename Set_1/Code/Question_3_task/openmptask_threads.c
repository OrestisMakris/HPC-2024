#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 1000  // Block size (grain size) for tasks

//--------------------------------------
// Heavy computation function
//--------------------------------------
double work(int i) {
    double result = 0.0;
    for (int j = 0; j < 10000; j++) { // Heavy inner loop
        result += sin(i * j) * cos(j / (i + 1.0)) + sqrt(i + j + 1.0);
    }
    return result;
}

//--------------------------------------
// Function Prototypes
//--------------------------------------
void initialize_serial(double *A, int size);
void initialize_threads(double *A, int size);
void initialize_tasks(double *A, int size);
void initialize_tasks_optimized(double *A, int size, int block_size);
void initialize_tasks_taskloop(double *A, int size, int block_size);
double compute_sum_tasks(int size, int block_size);
double max_difference(double *A, double *B, int size);

//--------------------------------------
// Serial implementation
//--------------------------------------
void initialize_serial(double *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

//--------------------------------------
// OpenMP threads implementation using parallel for
//--------------------------------------
void initialize_threads(double *A, int size) {
    #pragma omp parallel for schedule(dynamic, 2)
    for (int i = 0; i < size; i++) {
        A[i] = work(i);
    }
}

//--------------------------------------
// OpenMP tasks implementation: one task per iteration
//--------------------------------------
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
            #pragma omp taskwait
        }
    }
}

//--------------------------------------
// OpenMP Optimized Tasks:
// Group iterations into blocks to reduce task creation overhead.
// The 'untied' clause gives flexibility in task scheduling.
//--------------------------------------
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

//--------------------------------------
// OpenMP Taskloop Implementation:
// Uses the 'taskloop' directive to automatically create tasks.
//--------------------------------------
void initialize_tasks_taskloop(double *A, int size, int block_size) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp taskloop grainsize(block_size)
            for (int i = 0; i < size; i++) {
                A[i] = work(i);
            }
        }
    }
}

//--------------------------------------
// OpenMP Tasks with Reduction:
// Computes a global sum over all work(i) values using a taskloop with reduction.
// (Requires OpenMP 5.0 or later.)
//--------------------------------------
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

//--------------------------------------
// Utility: Compute maximum absolute difference between two arrays
//--------------------------------------
double max_difference(double *A, double *B, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(A[i] - B[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

//--------------------------------------
// Main: Run multiple tests over various array sizes and thread counts
//--------------------------------------
int main() {
    // Define the array sizes (number of iterations) to test.
    int sizes[] = {1000, 10000, 10000, 100000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Define the number of threads to test.
    int thread_counts[] = {2, 4, 8, 16, 24, 28, 32};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Open a CSV file for writing results.
    FILE *fp = fopen("results.csv", "w");
    if (fp == NULL) {
        perror("Cannot open results.csv for writing");
        return 1;
    }
    // CSV header: Method,ArraySize,NumThreads,ExecutionTime,MaxError,Sum
    fprintf(fp, "Method,ArraySize,NumThreads,ExecutionTime,MaxError,Sum\n");

    // Disable dynamic adjustment of threads.
    omp_set_dynamic(0);

    // For each array size, run tests.
    for (int s = 0; s < num_sizes; s++) {
        int current_size = sizes[s];
        printf("===== Array Size: %d =====\n", current_size);

        // Allocate the serial result array.
        double *A_serial = (double *)malloc(current_size * sizeof(double));
        if (A_serial == NULL) {
            fprintf(stderr, "Memory allocation failed for A_serial\n");
            return 1;
        }

        // --- Serial Test (baseline) ---
        omp_set_num_threads(1);
        double start = omp_get_wtime();
        initialize_serial(A_serial, current_size);
        double end = omp_get_wtime();
        double time_serial = end - start;
        printf("Serial: ArraySize = %d, Threads = %d, Time = %f sec\n", current_size, 1, time_serial);
        fprintf(fp, "Serial,%d,%d,%f,0,\n", current_size, 1, time_serial);

        // For each thread count, run the parallel methods.
        for (int t = 0; t < num_thread_counts; t++) {
            int nthreads = thread_counts[t];
            omp_set_num_threads(nthreads);

            // 1. Threads (parallel for)
            double *A_threads = (double *)malloc(current_size * sizeof(double));
            if (A_threads == NULL) {
                fprintf(stderr, "Memory allocation failed for A_threads\n");
                return 1;
            }
            start = omp_get_wtime();
            initialize_threads(A_threads, current_size);
            end = omp_get_wtime();
            double time_threads = end - start;
            double err_threads = max_difference(A_serial, A_threads, current_size);
            printf("Threads: ArraySize = %d, Threads = %d, Time = %f sec, MaxDiff = %e\n",
                   current_size, nthreads, time_threads, err_threads);
            fprintf(fp, "Threads,%d,%d,%f,%e,\n", current_size, nthreads, time_threads, err_threads);
            free(A_threads);

            // 2. Tasks (one task per iteration)
            double *A_tasks = (double *)malloc(current_size * sizeof(double));
            if (A_tasks == NULL) {
                fprintf(stderr, "Memory allocation failed for A_tasks\n");
                return 1;
            }
            start = omp_get_wtime();
            initialize_tasks(A_tasks, current_size);
            end = omp_get_wtime();
            double time_tasks = end - start;
            double err_tasks = max_difference(A_serial, A_tasks, current_size);
            printf("Tasks: ArraySize = %d, Threads = %d, Time = %f sec, MaxDiff = %e\n",
                   current_size, nthreads, time_tasks, err_tasks);
            fprintf(fp, "Tasks,%d,%d,%f,%e,\n", current_size, nthreads, time_tasks, err_tasks);
            free(A_tasks);

            // 3. Taskloop
            double *A_taskloop = (double *)malloc(current_size * sizeof(double));
            if (A_taskloop == NULL) {
                fprintf(stderr, "Memory allocation failed for A_taskloop\n");
                return 1;
            }
            start = omp_get_wtime();
            initialize_tasks_taskloop(A_taskloop, current_size, BLOCK_SIZE);
            end = omp_get_wtime();
            double time_taskloop = end - start;
            double err_taskloop = max_difference(A_serial, A_taskloop, current_size);
            printf("Taskloop: ArraySize = %d, Threads = %d, Time = %f sec, MaxDiff = %e\n",
                   current_size, nthreads, time_taskloop, err_taskloop);
            fprintf(fp, "Taskloop,%d,%d,%f,%e,\n", current_size, nthreads, time_taskloop, err_taskloop);
            free(A_taskloop);

            // 4. Optimized Tasks (grouped tasks)
            double *A_tasks_opt = (double *)malloc(current_size * sizeof(double));
            if (A_tasks_opt == NULL) {
                fprintf(stderr, "Memory allocation failed for A_tasks_opt\n");
                return 1;
            }
            start = omp_get_wtime();
            initialize_tasks_optimized(A_tasks_opt, current_size, BLOCK_SIZE);
            end = omp_get_wtime();
            double time_tasks_opt = end - start;
            double err_tasks_opt = max_difference(A_serial, A_tasks_opt, current_size);
            printf("OptimizedTasks: ArraySize = %d, Threads = %d, Time = %f sec, MaxDiff = %e\n",
                   current_size, nthreads, time_tasks_opt, err_tasks_opt);
            fprintf(fp, "OptimizedTasks,%d,%d,%f,%e,\n", current_size, nthreads, time_tasks_opt, err_tasks_opt);
            free(A_tasks_opt);

            // 5. Tasks Reduction (global sum)
            start = omp_get_wtime();
            double sum_tasks = compute_sum_tasks(current_size, BLOCK_SIZE);
            end = omp_get_wtime();
            double time_reduction = end - start;
            printf("Reduction: ArraySize = %d, Threads = %d, Time = %f sec, Sum = %f\n",
                   current_size, nthreads, time_reduction, sum_tasks);
            fprintf(fp, "Reduction,%d,%d,%f,,%f\n", current_size, nthreads, time_reduction, sum_tasks);
        }
        printf("\n");
        free(A_serial);
    }

    fclose(fp);
    printf("Results saved to results.csv\n");
    return 0;
}
