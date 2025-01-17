
//***********************************************************
//  University of Patras
//      Department of Computer Emgineering & Informatics
//
//      Course: Parallel Programming in AI
//      Week 09: OpenMP - GPU programming
//
//  Description: Matrix Multiplication/Addition  C = A^N
//      Author: Evangelos Dermatas
//      v1.0


// ***************************************************************
// ******************************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// ******************************************************************************************
#define SIZE 1024 // Size of the square matrix

float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

// ******************************************************************************************
void initRandMatrix(float matrix[SIZE][SIZE]) {
// #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (float) (rand() % 400 ) / 100.0f - 2.0f ; // Random numbers in [-2,2]
        }
    }
}


// ******************************************************************************************
void initMatrix(float matrix[SIZE][SIZE], float Val) {

#pragma omp target data use_device_addr(matrix)
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = Val ; 
        }
    }
}

// ******************************************************************************************
void copyMatrix(float m1[SIZE][SIZE],float m2[SIZE][SIZE] ) {

#pragma omp target data use_device_addr(m1,m2)
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m1[i][j] = m2[i][j] ; 
        }
    }
}

// ******************************************************************************************
int main(int argc, char *argv[]) {
int N = 10 ;


	if ( argc >= 2 )
		N = atoi(argv[1]) ;

	if ( N < 2 ) 
		N = 2 ;

    initRandMatrix(A);

#pragma omp target data map(to: A) map(from: C) map(alloc:B)
{
	initMatrix(B,0) ;

#pragma omp target teams distribute parallel for collapse(2)
//#pragma omp target loop
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
				float f = 0 ;
                for (int k = 0; k < SIZE; k++) {
                    f += A[i][k] * A[k][j];
                }
                B[i][j] = f ;
            }
        }

	for (int r = 1; r < N; r++) {
		initMatrix(C,0) ;
#pragma omp target teams distribute parallel for collapse(2)
//#pragma omp target loop
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
            	float f = 0 ;
                for (int k = 0; k < SIZE; k++) {
                    f += B[i][k] * A[k][j];
                }
                C[i][j]=f ;
            }
        }
        copyMatrix( B, C) ;
    }
}
	// Print three numbers of C
    printf("Matrix C: %f %f %f", C[0][0], C[0][1], C[1][1]);
    return 0;
}
