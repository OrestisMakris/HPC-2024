//************************************************************
//  University of Patras
//      Department of Computer Emgineering & Informatics
//
//      Course: Parallel Programming in AI
//      Week 09: OpenMP - GPU programming
//
//  Description: change Temperature in a plate
//      Author: Evangelos Dermatas
//      v1.0


// ***************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// ***************************************************************

#define N	1000
#define M	1000
#define GG	(1.0f/6.0f)

//float A[N][M] ;
//float B[N][M] ;

#pragma omp begin declare target
float A[N][M] ;
float B[N][M] ;
#pragma omp end declare target

// ***************************************************************
void initTemp(void) {

float t = rand()%100 - 50 ;
#pragma omp parallel for simd
for( int i = 0 ; i < N ; i++ )
        A[i][0] = t ;
#pragma omp parallel for simd
for( int i = 0 ; i < N ; i++ )
        B[i][0] = t ;

t = rand()%100 - 50 ;
#pragma omp parallel for simd
for( int i = 0 ; i < N ; i++ )
        A[i][M-1] = t ;
#pragma omp parallel for simd
for( int i = 0 ; i < N ; i++ )
        B[i][M-1] = t ;

t = rand()%100 - 50 ;
#pragma omp parallel for simd
for( int j = 0 ; j < M ; j++ )
        A[0][j] = t ;
#pragma omp parallel for simd
for( int j = 0 ; j < M ; j++ )
        B[0][j] = t ;

t = rand()%100 - 50 ;
#pragma omp parallel for simd
for( int j = 0 ; j < M ; j++ )
        A[N-1][j] = t ;
#pragma omp parallel for simd
for( int j = 0 ; j < M ; j++ )
        B[N-1][j] = t ;

t = rand()%100 - 50 ;
#pragma omp parallel for simd collapse(2)
for( int i = 1 ; i < N-1 ; i++ )
for( int j = 1 ; j < M-1 ; j++ )
        A[i][j] = t ;
}


// ***************************************************************
#pragma omp begin declare target
float computeTemp(int Time) {

    float sum = 0.0;

    // Offload to the GPU
    for (int k = 0; k < Time; k++) {

			#pragma omp teams distribute parallel for collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < M - 1; j++)
                    B[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1] + 2 * A[i][j]) * GG;
            }

            #pragma omp teams distribute parallel for collapse(2)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < M - 1; j++)
                    A[i][j] = (B[i - 1][j] + B[i + 1][j] + B[i][j - 1] + B[i][j + 1] + 2 * B[i][j]) * GG;
            }
        }

        #pragma omp teams distribute parallel for reduction(+:sum) collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                sum += A[i][j];
            }
        }

    return sum;
}
#pragma omp end declare target


// ***************************************************************
int main(int argc, char *argv[]) {
	float t ;
	int rep ;

	printf( "CPU\n" ) ;
	initTemp() ;
	printf( "GPU\n" ) ;

	if ( argc == 1 )
		rep = 1000 ;
	else
		rep = atoi(argv[1]) ;

#pragma omp target enter data map(to:A[0:N][0:N], B[0:N][0:N], rep)

	t = computeTemp(rep) ;

#pragma omp target exit data map(delete:A[0:N][0:N], B[0:N][0:N])

	printf( "Sum=%f", t ) ;
	return 0;
}
