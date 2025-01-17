// *******************************************************************************
#include <stdio.h>
#include <omp.h>

// *******************************************************************************
#define N 10000

// Declare and allocate arrays
float A[N][N], B[N][N], C[N][N];

// *******************************************************************************
void initArrays( float Ar[N][N], float Ar2[N][N] ) {

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ar[i][j] = i * 1.0f + j;
            Ar2[i][j] = i * 2.0f +j;
        }
	}
}

// *******************************************************************************
void initArray( float Ar[N][N] ) {
//#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ar[i][j] = i * 1.5f + j;
        }
	}
}




// *******************************************************************************
int main() {

    printf( "CPU\n" ) ;
    initArray(A) ;
    // Print the top left 2x2 corner of the matrix
    printf("Matrix A:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    initArray(B) ;
    // Print the top left 2x2 corner of the matrix
    printf("Matrix B:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", B[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf( "GPU\n" ) ;
    // Offload the computation to the GPU
    #pragma omp target map(to: A[0:N][0:N], B[0:N][0:N]) map(from: C[0:N][0:N])
	{
        #pragma omp teams distribute parallel for simd collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
		}
	}

    printf("Matrix C:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
