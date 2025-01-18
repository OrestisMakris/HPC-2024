#pragma once
#include <immintrin.h>
#include <omp.h>

static inline float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{
         // Smoothness indicators, which measure the oscillation of the solution
		const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
		const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
		const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);

		const float is0plus = is0 + (float)WENOEPS;
		const float is1plus = is1 + (float)WENOEPS;
		const float is2plus = is2 + (float)WENOEPS;

		const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
		const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
		const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
		const float alphasum = alpha0+alpha1+alpha2;
		const float inv_alpha = ((float)1)/alphasum;

		const float omega0 = alpha0 * inv_alpha;
		const float omega1 = alpha1 * inv_alpha;
		const float omega2 = 1-omega0-omega1;

		return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
					 omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
					 omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);
}


// WENO reconstruction for the minus flux
void weno_minus_reference(const float * const a, const float * const b, const float * const c,
			  const float * const d, const float * const e, float * const out,
			  const int NENTRIES)
{
//#pragma omp for
		for (int i=0; i<NENTRIES; ++i)
			out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}


// Auto-vectorized version of weno_minus implementation

void weno_minus_auto(const float * const a, const float * const b, const float * const c,
			  const float * const d, const float * const e, float * const out,
			  const int NENTRIES)
{	
	#pragma GCC ivdep
	for (int i=0; i<NENTRIES; ++i){
		out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
	}
}

void weno_minus_unrolled(const float * const a, const float * const b, const float * const c,
                         const float * const d, const float * const e, float * const out,
                         const int NENTRIES)
{
    int i = 0;
    const int unroll_factor = 4; // Process 4 elements per iteration
    int limit = NENTRIES / unroll_factor * unroll_factor;

    #pragma GCC ivdep
    for (; i < limit; i += unroll_factor) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
        out[i + 1] = weno_minus_core(a[i + 1], b[i + 1], c[i + 1], d[i + 1], e[i + 1]);
        out[i + 2] = weno_minus_core(a[i + 2], b[i + 2], c[i + 2], d[i + 2], e[i + 2]);
        out[i + 3] = weno_minus_core(a[i + 3], b[i + 3], c[i + 3], d[i + 3], e[i + 3]);
    }

    // Handle any remaining elements
    for (; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}


void weno_minus_builtin(const float * const a, const float * const b, const float * const c,
                        const float * const d, const float * const e, float * const out,
                        const int NENTRIES)
{
    // Assume the arrays are 32-byte aligned for vectorized access
    const float * a_aligned = __builtin_assume_aligned(a, 32);
    const float * b_aligned = __builtin_assume_aligned(b, 32);
    const float * c_aligned = __builtin_assume_aligned(c, 32);
    const float * d_aligned = __builtin_assume_aligned(d, 32);
    const float * e_aligned = __builtin_assume_aligned(e, 32);
    float * out_aligned = __builtin_assume_aligned(out, 32);

    #pragma GCC ivdep
    for (int i = 0; i < NENTRIES; ++i) {
        out_aligned[i] = weno_minus_core(a_aligned[i], b_aligned[i], c_aligned[i], d_aligned[i], e_aligned[i]);
    }
}


// OpenMP SIMD implementation
void weno_minus_omp(const float * const a, const float * const b, const float * const c,
                   const float * const d, const float * const e, float * const out,
                   const int NENTRIES)
{
    #pragma omp simd
    for (int i = 0; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}



// // AVX implementation
// void weno_minus_avx(const float * const a, const float * const b, const float * const c,
//                    const float * const d, const float * const e, float * const out,
//                    const int NENTRIES)
// {
//     // Process 8 elements at a time using AVX
//     const int vec_size = NENTRIES - (NENTRIES % 8);
    
//     for (int i = 0; i < vec_size; i += 8) {
//         // Load 8 elements from each array
//         __m256 va = _mm256_load_ps(&a[i]);
//         __m256 vb = _mm256_load_ps(&b[i]);
//         __m256 vc = _mm256_load_ps(&c[i]);
//         __m256 vd = _mm256_load_ps(&d[i]);
//         __m256 ve = _mm256_load_ps(&e[i]);
        
//         // Constants
//         const __m256 v4_3 = _mm256_set1_ps(4.0f/3.0f);
//         const __m256 v19_3 = _mm256_set1_ps(19.0f/3.0f);
//         const __m256 v11_3 = _mm256_set1_ps(11.0f/3.0f);
//         const __m256 v25_3 = _mm256_set1_ps(25.0f/3.0f);
//         const __m256 v31_3 = _mm256_set1_ps(31.0f/3.0f);
//         const __m256 v10_3 = _mm256_set1_ps(10.0f/3.0f);
//         const __m256 v13_3 = _mm256_set1_ps(13.0f/3.0f);
//         const __m256 v5_3 = _mm256_set1_ps(5.0f/3.0f);
        
//         // Calculate smoothness indicators (is0, is1, is2)
//         __m256 is0 = _mm256_mul_ps(va, _mm256_fmadd_ps(va, v4_3,
//                     _mm256_fmadd_ps(vb, _mm256_set1_ps(-19.0f/3.0f),
//                     _mm256_mul_ps(vc, v11_3))));
//         is0 = _mm256_add_ps(is0, _mm256_mul_ps(vb, _mm256_fmadd_ps(vb, v25_3,
//                     _mm256_mul_ps(vc, _mm256_set1_ps(-31.0f/3.0f)))));
//         is0 = _mm256_add_ps(is0, _mm256_mul_ps(vc, _mm256_mul_ps(vc, v10_3)));

//         // Similar calculations for is1 and is2...
//         // (Truncated for brevity - full implementation would include all calculations)

//         // Store results
//         _mm256_store_ps(&out[i], is0); // This is simplified - full implementation would complete the WENO calculation
//     }

//     // Handle remaining elements
//     for (int i = vec_size; i < NENTRIES; ++i) {
//         out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
//     }
// }