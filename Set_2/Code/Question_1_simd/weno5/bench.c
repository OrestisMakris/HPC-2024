#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"

float * myalloc(const int NENTRIES, const int verbose )
{
	const int initialize = 1;
	enum { alignment_bytes = 32 } ;
	float * tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);

	if (initialize)
	{
		for(int i=0; i<NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for(int i=0; i<NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t,  NULL);
	return t.tv_sec + t.tv_usec*1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
    static const int verbose = 0;
    int failed = 0;

    for (int i = 0; i < N; ++i)
    {
        assert(!isnan(ref[i]));
        assert(!isnan(val[i]));

        const double err = ref[i] - val[i];
        const double relerr = err / fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

        if (verbose) printf("+%1.1e,", relerr);

        if (fabs(relerr) >= tol && fabs(err) >= tol)
        {
            // Report the failing index with error information
            printf("\nERROR: %d: ref = %e, val = %e, abs_err = %e, rel_err = %e\n",
                   i, ref[i], val[i], err, relerr);
            failed = 1;
        }

        // Assert that the error is within tolerance
        assert(fabs(relerr) < tol || fabs(err) < tol);
    }

    if (verbose) printf("\t");

    // If any error was found, terminate the program
    if (failed)
    {
        printf("Accuracy check failed! Please check the error details above.\n");
        exit(EXIT_FAILURE);
    }
}

// Function pointer type for WENO implementations
typedef void (*weno_func)(const float * const, const float * const,
                         const float * const, const float * const,
                         const float * const, float * const, const int);

void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name, weno_func func)
{
	const int NENTRIES = 4 * (NENTRIES_ / 4);

	printf("Testing with %e entries\n", (float)NENTRIES);

	float * const a = myalloc(NENTRIES, verbose);
	float * const b = myalloc(NENTRIES, verbose);
	float * const c = myalloc(NENTRIES, verbose);
	float * const d = myalloc(NENTRIES, verbose);
	float * const e = myalloc(NENTRIES, verbose);
	//float * const f = myalloc(NENTRIES, verbose);
	float * const gold = myalloc(NENTRIES, verbose);
	float * const result = myalloc(NENTRIES, verbose);

 // Time for weno_minus_reference for gold
    double start_time = get_wtime();
    weno_minus_reference(a, b, c, d, e, gold, NENTRIES);
    double end_time = get_wtime();
    printf("Time for weno_minus_reference (gold): %.6f seconds\n", end_time - start_time);

    // Time for different implementations
    start_time = get_wtime();
    func(a, b, c, d, e, result, NENTRIES);
    end_time = get_wtime();
    printf("%s implementation: %.3f seconds\n", benchmark_name, end_time - start_time);

	const double tol = 1e-5;
	printf("minus: verifying accuracy with tolerance %.5e...", tol);
	check_error(tol, gold, result, NENTRIES);
	printf("passed!\n");

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}


int main (int argc, char *  argv[])
{
	printf("Hello, weno benchmark!\n");
	const int debug = 1;

	int verbose = 0;
	int NENTRIES = 2000e6;
	int NTIMES = 1;

  // Determine which implementation to use based on binary name
    weno_func implementation = weno_minus_reference;
    char *impl_name = "Reference";
    
    const char *binary_name = strrchr(argv[0], '/');
    if (binary_name == NULL) {
        binary_name = argv[0];
    } else {
        binary_name++; // Skip the '/'
    }
    
    if (strcmp(binary_name, "bench_auto") == 0) {
        implementation = weno_minus_auto;
        impl_name = "Auto-vectorized";
    } else if (strcmp(binary_name, "bench_omp") == 0) {
        implementation = weno_minus_omp;
        impl_name = "OpenMP SIMD";
    // } else if (strcmp(binary_name, "bench_simd") == 0) {
    //     implementation = weno_minus_avx;
    //     impl_name = "AVX";
    }
	

	if (strcmp(binary_name, "bench_builtin") == 0) {
    implementation = weno_minus_builtin;
    impl_name = "Builtin-aligned";
	}

	// if (strcmp(binary_name, "bench_unrolled") == 0) {
	// implementation = weno_minus_unrolled;
	// impl_name = "Unrolled";
	// }

	if (strcmp(binary_name, "bench_omp_optimized") == 0) {
	implementation = weno_minus_omp_optimized;
	impl_name = "OpenMP SIMD Optimized";
	}

	printf("Running %s implementation\n", impl_name);

	if (debug)
	{
		benchmark(argc, argv, NENTRIES, NTIMES,verbose, impl_name, implementation);
		return 0;
	}

	/* performance on cache hits */
	{
		//const double desired_kb =  16 * 4 * 0.5; /* we want to fill 50% of the dcache */
		const int nentries =  16 * (int)(pow(32 + 6, 2) * 4);//floor(desired_kb * 1024. / 7 / sizeof(float));
		const int ntimes = (int)floor(2. / (1e-7 * nentries));

		for(int i=0; i<4; ++i)
		{
			printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, ntimes, 0, impl_name, implementation);
		}
	}

	/* performance on data streams */
	{
		const double desired_mb =  128 * 4;
		const int nentries =  (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

		for(int i=0; i<4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, 1, 0, impl_name, implementation);
		}
	}

    return 0;
}
