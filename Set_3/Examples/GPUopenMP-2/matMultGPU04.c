
//************************************************************
//  University of Patras
//      Department of Computer Emgineering & Informatics
//
//      Course: Parallel Programming in AI
//      Week 09: OpenMP - GPU programming
//
//  Description: Matrix Multiplication
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
// Function to initialize matrices
void initMatrix(float matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (float) (rand() % 400 ) / 100.0f - 2.0f ; // Random numbers in [-2,2]
        }
    }
}

// ******************************************************************************************
int main() {

    // Initialize matrices A and B
    initMatrix(A);
    initMatrix(B);

	// Compute C=A*B
    #pragma omp target data map(to: A, B) map(from: C)
    #pragma omp target loop
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                for (int k = 0; k < SIZE; k++)
                    C[i][j]+= A[i][k] * B[k][j];

	// Print three numbers of C
    printf("Matrix C: %f %f %f", C[0][0], C[0][1], C[1][1]);

    return 0;
}
