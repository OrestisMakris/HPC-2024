
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
#define SIZE 4 // Size of the square matrix

float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

// ******************************************************************************************
// Function to initialize matrices
void initMatrix(float matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (float) (rand() % 100 ) / 10.0f; // Random numbers in [0,9]
        }
    }
}

// ******************************************************************************************
int main() {

    // Initialize matrices A and B
    initMatrix(A);
    // Print A
    printf("Matrix A:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%.2f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    initMatrix(B);
    // Print B
    printf("Matrix B:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%.2f ", B[i][j]);
        }
        printf("\n");
    }
    printf("\n");

	// Compute C=A*B
    #pragma omp target data map(to: A, B) map(from: C)
    {
        #pragma omp target parallel for
        for (int i = 0; i < SIZE; i++){
            for (int j = 0; j < SIZE; j++){
                for (int k = 0; k < SIZE; k++){
                    C[i][j]+= A[i][k] * B[k][j];
                }
            }
        }
    }
    

	// Print C
    printf("Matrix C:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%.2f\t", C[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
