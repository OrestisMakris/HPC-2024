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
    #pragma omp for simd aligned(a, b, c, d, e, out: 32) 
    for (int i = 0; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}


void weno_minus_omp_optimized(const float * restrict a, const float * restrict b, const float * restrict c,
                              const float * restrict d, const float * restrict e, float * restrict out,
                              const int NENTRIES)
{
    // Ensure NENTRIES is a multiple of vector length for efficient SIMD
    const int vector_width = 8; // For AVX, change to 16 for AVX-512
    const int simd_limit = NENTRIES / vector_width * vector_width;

    #pragma omp parallel for simd aligned(a, b, c, d, e, out: 32) schedule(static)
    for (int i = 0; i < simd_limit; i += vector_width) {
        #pragma omp simd aligned(a, b, c, d, e, out: 32)
        for (int j = 0; j < vector_width; ++j) {
            int idx = i + j;
            out[idx] = weno_minus_core(a[idx], b[idx], c[idx], d[idx], e[idx]);
        }
    }

    // Process remaining entries if NENTRIES is not a multiple of vector width
    for (int i = simd_limit; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}


// AVX implementation
void weno_minus_avx(const float *restrict a, const float *restrict b,
                    const float *restrict c, const float *restrict d,
                    const float *restrict e, float *restrict out,
                    const int NENTRIES)
{
    int i;
    for (i = 0; i <= NENTRIES - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 vd = _mm256_loadu_ps(&d[i]);
        __m256 ve = _mm256_loadu_ps(&e[i]);

        // is0 calculation matching reference order
        __m256 temp1_is0 = _mm256_mul_ps(va, _mm256_set1_ps(4.0f/3.0f));
        temp1_is0 = _mm256_fmadd_ps(vb, _mm256_set1_ps(-19.0f/3.0f), temp1_is0);
        temp1_is0 = _mm256_fmadd_ps(vc, _mm256_set1_ps(11.0f/3.0f), temp1_is0);
        temp1_is0 = _mm256_mul_ps(va, temp1_is0);

        __m256 temp2_is0 = _mm256_mul_ps(vb, _mm256_set1_ps(25.0f/3.0f));
        temp2_is0 = _mm256_fmadd_ps(vc, _mm256_set1_ps(-31.0f/3.0f), temp2_is0);
        temp2_is0 = _mm256_mul_ps(vb, temp2_is0);

        __m256 temp3_is0 = _mm256_mul_ps(vc, vc);
        temp3_is0 = _mm256_mul_ps(temp3_is0, _mm256_set1_ps(10.0f/3.0f));

        __m256 is0 = _mm256_add_ps(temp1_is0, temp2_is0);
        is0 = _mm256_add_ps(is0, temp3_is0);

        // is1 calculation matching reference order
        __m256 temp1_is1 = _mm256_mul_ps(vb, _mm256_set1_ps(4.0f/3.0f));
        temp1_is1 = _mm256_fmadd_ps(vc, _mm256_set1_ps(-13.0f/3.0f), temp1_is1);
        temp1_is1 = _mm256_fmadd_ps(vd, _mm256_set1_ps(5.0f/3.0f), temp1_is1);
        temp1_is1 = _mm256_mul_ps(vb, temp1_is1);

        __m256 temp2_is1 = _mm256_mul_ps(vc, _mm256_set1_ps(13.0f/3.0f));
        temp2_is1 = _mm256_fmadd_ps(vd, _mm256_set1_ps(-13.0f/3.0f), temp2_is1);
        temp2_is1 = _mm256_mul_ps(vc, temp2_is1);

        __m256 temp3_is1 = _mm256_mul_ps(vd, vd);
        temp3_is1 = _mm256_mul_ps(temp3_is1, _mm256_set1_ps(4.0f/3.0f));

        __m256 is1 = _mm256_add_ps(temp1_is1, temp2_is1);
        is1 = _mm256_add_ps(is1, temp3_is1);

        // is2 calculation matching reference order
        __m256 temp1_is2 = _mm256_mul_ps(vc, _mm256_set1_ps(10.0f/3.0f));
        temp1_is2 = _mm256_fmadd_ps(vd, _mm256_set1_ps(-31.0f/3.0f), temp1_is2);
        temp1_is2 = _mm256_fmadd_ps(ve, _mm256_set1_ps(11.0f/3.0f), temp1_is2);
        temp1_is2 = _mm256_mul_ps(vc, temp1_is2);

        __m256 temp2_is2 = _mm256_mul_ps(vd, _mm256_set1_ps(25.0f/3.0f));
        temp2_is2 = _mm256_fmadd_ps(ve, _mm256_set1_ps(-19.0f/3.0f), temp2_is2);
        temp2_is2 = _mm256_mul_ps(vd, temp2_is2);

        __m256 temp3_is2 = _mm256_mul_ps(ve, ve);
        temp3_is2 = _mm256_mul_ps(temp3_is2, _mm256_set1_ps(4.0f/3.0f));

        __m256 is2 = _mm256_add_ps(temp1_is2, temp2_is2);
        is2 = _mm256_add_ps(is2, temp3_is2);

        // Add WENOEPS exactly as in reference
        __m256 is0plus = _mm256_add_ps(is0, _mm256_set1_ps(WENOEPS));
        __m256 is1plus = _mm256_add_ps(is1, _mm256_set1_ps(WENOEPS));
        __m256 is2plus = _mm256_add_ps(is2, _mm256_set1_ps(WENOEPS));

        // Calculate alphas matching reference ordering
        __m256 is0plus_sq = _mm256_mul_ps(is0plus, is0plus);
        __m256 is1plus_sq = _mm256_mul_ps(is1plus, is1plus);
        __m256 is2plus_sq = _mm256_mul_ps(is2plus, is2plus);

        __m256 alpha0 = _mm256_div_ps(_mm256_set1_ps(1.0f), is0plus_sq);
        alpha0 = _mm256_mul_ps(_mm256_set1_ps(0.1f), alpha0);

        __m256 alpha1 = _mm256_div_ps(_mm256_set1_ps(1.0f), is1plus_sq);
        alpha1 = _mm256_mul_ps(_mm256_set1_ps(0.6f), alpha1);

        __m256 alpha2 = _mm256_div_ps(_mm256_set1_ps(1.0f), is2plus_sq);
        alpha2 = _mm256_mul_ps(_mm256_set1_ps(0.3f), alpha2);

        // Calculate alphasum and omega weights matching reference
        __m256 alphasum = _mm256_add_ps(alpha0, alpha1);
        alphasum = _mm256_add_ps(alphasum, alpha2);
        __m256 inv_alpha = _mm256_div_ps(_mm256_set1_ps(1.0f), alphasum);

        __m256 omega0 = _mm256_mul_ps(alpha0, inv_alpha);
        __m256 omega1 = _mm256_mul_ps(alpha1, inv_alpha);
        __m256 omega_sum = _mm256_add_ps(omega0, omega1);
        __m256 omega2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), omega_sum);

        // Final calculation matching reference order
        __m256 term0 = _mm256_mul_ps(va, _mm256_set1_ps(1.0f/3.0f));
        term0 = _mm256_fmadd_ps(vb, _mm256_set1_ps(-7.0f/6.0f), term0);
        term0 = _mm256_fmadd_ps(vc, _mm256_set1_ps(11.0f/6.0f), term0);
        term0 = _mm256_mul_ps(omega0, term0);

        __m256 term1 = _mm256_mul_ps(vb, _mm256_set1_ps(-1.0f/6.0f));
        term1 = _mm256_fmadd_ps(vc, _mm256_set1_ps(5.0f/6.0f), term1);
        term1 = _mm256_fmadd_ps(vd, _mm256_set1_ps(1.0f/3.0f), term1);
        term1 = _mm256_mul_ps(omega1, term1);

        __m256 term2 = _mm256_mul_ps(vc, _mm256_set1_ps(1.0f/3.0f));
        term2 = _mm256_fmadd_ps(vd, _mm256_set1_ps(5.0f/6.0f), term2);
        term2 = _mm256_fmadd_ps(ve, _mm256_set1_ps(-1.0f/6.0f), term2);
        term2 = _mm256_mul_ps(omega2, term2);

        __m256 result = _mm256_add_ps(term0, term1);
        result = _mm256_add_ps(result, term2);

        // Store result
        _mm256_storeu_ps(&out[i], result);
    }

    // Handle remaining elements
    for (; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}

void weno_minus_sse(const float *restrict a, const float *restrict b,
                    const float *restrict c, const float *restrict d,
                    const float *restrict e, float *restrict out,
                    const int NENTRIES)
{
    int i;
    for (i = 0; i <= NENTRIES - 4; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_loadu_ps(&c[i]);
        __m128 vd = _mm_loadu_ps(&d[i]);
        __m128 ve = _mm_loadu_ps(&e[i]);

        // Smoothness indicators
        // is0 calculation
        __m128 is0 = _mm_add_ps(
            _mm_mul_ps(va, _mm_add_ps(
                _mm_mul_ps(va, _mm_set1_ps(4.0f/3.0f)),
                _mm_add_ps(
                    _mm_mul_ps(vb, _mm_set1_ps(-19.0f/3.0f)),
                    _mm_mul_ps(vc, _mm_set1_ps(11.0f/3.0f))))),
            _mm_add_ps(
                _mm_mul_ps(vb, _mm_add_ps(
                    _mm_mul_ps(vb, _mm_set1_ps(25.0f/3.0f)),
                    _mm_mul_ps(vc, _mm_set1_ps(-31.0f/3.0f)))),
                _mm_mul_ps(_mm_mul_ps(vc, vc), _mm_set1_ps(10.0f/3.0f))));

        // is1 calculation
        __m128 is1 = _mm_add_ps(
            _mm_mul_ps(vb, _mm_add_ps(
                _mm_mul_ps(vb, _mm_set1_ps(4.0f/3.0f)),
                _mm_add_ps(
                    _mm_mul_ps(vc, _mm_set1_ps(-13.0f/3.0f)),
                    _mm_mul_ps(vd, _mm_set1_ps(5.0f/3.0f))))),
            _mm_add_ps(
                _mm_mul_ps(vc, _mm_add_ps(
                    _mm_mul_ps(vc, _mm_set1_ps(13.0f/3.0f)),
                    _mm_mul_ps(vd, _mm_set1_ps(-13.0f/3.0f)))),
                _mm_mul_ps(_mm_mul_ps(vd, vd), _mm_set1_ps(4.0f/3.0f))));

        // is2 calculation
        __m128 is2 = _mm_add_ps(
            _mm_mul_ps(vc, _mm_add_ps(
                _mm_mul_ps(vc, _mm_set1_ps(10.0f/3.0f)),
                _mm_add_ps(
                    _mm_mul_ps(vd, _mm_set1_ps(-31.0f/3.0f)),
                    _mm_mul_ps(ve, _mm_set1_ps(11.0f/3.0f))))),
            _mm_add_ps(
                _mm_mul_ps(vd, _mm_add_ps(
                    _mm_mul_ps(vd, _mm_set1_ps(25.0f/3.0f)),
                    _mm_mul_ps(ve, _mm_set1_ps(-19.0f/3.0f)))),
                _mm_mul_ps(_mm_mul_ps(ve, ve), _mm_set1_ps(4.0f/3.0f))));

        // Add WENOEPS
        __m128 is0plus = _mm_add_ps(is0, _mm_set1_ps(WENOEPS));
        __m128 is1plus = _mm_add_ps(is1, _mm_set1_ps(WENOEPS));
        __m128 is2plus = _mm_add_ps(is2, _mm_set1_ps(WENOEPS));

        // Calculate squared denominators
        __m128 is0plus_sq = _mm_mul_ps(is0plus, is0plus);
        __m128 is1plus_sq = _mm_mul_ps(is1plus, is1plus);
        __m128 is2plus_sq = _mm_mul_ps(is2plus, is2plus);

        // Alpha weights
        __m128 alpha0 = _mm_div_ps(_mm_set1_ps(0.1f), is0plus_sq);
        __m128 alpha1 = _mm_div_ps(_mm_set1_ps(0.6f), is1plus_sq);
        __m128 alpha2 = _mm_div_ps(_mm_set1_ps(0.3f), is2plus_sq);

        // Normalize alpha weights
        __m128 alphasum = _mm_add_ps(_mm_add_ps(alpha0, alpha1), alpha2);
        __m128 inv_alpha = _mm_div_ps(_mm_set1_ps(1.0f), alphasum);

        // Calculate omega weights
        __m128 omega0 = _mm_mul_ps(alpha0, inv_alpha);
        __m128 omega1 = _mm_mul_ps(alpha1, inv_alpha);
        // omega2 = 1 - omega0 - omega1 for better numerical stability
        __m128 omega2 = _mm_sub_ps(_mm_set1_ps(1.0f), 
                                  _mm_add_ps(omega0, omega1));

        // Compute stencil values
        __m128 stencil0 = _mm_add_ps(
            _mm_mul_ps(va, _mm_set1_ps(1.0f/3.0f)),
            _mm_add_ps(
                _mm_mul_ps(vb, _mm_set1_ps(-7.0f/6.0f)),
                _mm_mul_ps(vc, _mm_set1_ps(11.0f/6.0f))));

        __m128 stencil1 = _mm_add_ps(
            _mm_mul_ps(vb, _mm_set1_ps(-1.0f/6.0f)),
            _mm_add_ps(
                _mm_mul_ps(vc, _mm_set1_ps(5.0f/6.0f)),
                _mm_mul_ps(vd, _mm_set1_ps(1.0f/3.0f))));

        __m128 stencil2 = _mm_add_ps(
            _mm_mul_ps(vc, _mm_set1_ps(1.0f/3.0f)),
            _mm_add_ps(
                _mm_mul_ps(vd, _mm_set1_ps(5.0f/6.0f)),
                _mm_mul_ps(ve, _mm_set1_ps(-1.0f/6.0f))));

        // Compute final result
        __m128 result = _mm_add_ps(
            _mm_mul_ps(omega0, stencil0),
            _mm_add_ps(
                _mm_mul_ps(omega1, stencil1),
                _mm_mul_ps(omega2, stencil2)));

        // Store result
        _mm_storeu_ps(&out[i], result);
    }

    // Handle remaining elements
    for (; i < NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}