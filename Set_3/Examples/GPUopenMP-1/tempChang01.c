//************************************************************
//  University of Patras
//      Department of Computer Emgineering & Informatics
//
//      Course: Parallel Programming in AI
//      Week 09: OpenMP - GPU programming
//
//  Description: change Temperature in a plate - Serial Version
//      Author: Evangelos Dermatas
//      v1.0


// ***************************************************************
#include <stdio.h>
#include <stdlib.h>
// ***************************************************************

#define N	1000
#define M	1000

float A[N][M] ;
float B[N][M] ;

// ***************************************************************
void initTemp(void)
{

float t = rand()%100 - 50 ;
for( int i = 0 ; i < N ; i++ )
        A[i][0] = t ;
for( int i = 0 ; i < N ; i++ )
        B[i][0] = t ;

t = rand()%100 - 50 ;
for( int i = 0 ; i < N ; i++ )
        A[i][M-1] = t ;
for( int i = 0 ; i < N ; i++ )
        B[i][M-1] = t ;

t = rand()%100 - 50 ;
for( int j = 0 ; j < M ; j++ )
        A[0][j] = t ;
for( int j = 0 ; j < M ; j++ )
        B[0][j] = t ;

t = rand()%100 - 50 ;
for( int j = 0 ; j < M ; j++ )
        A[N-1][j] = t ;
for( int j = 0 ; j < M ; j++ )
        B[N-1][j] = t ;

t = rand()%100 - 50 ;
for( int i = 1 ; i < N-1 ; i++ )
for( int j = 1 ; j < M-1 ; j++ )
        A[i][j] = t ;
}


// ***************************************************************
float computeTemp( int Time )
{

for(int k = 0 ; k< Time ; k++ ) {

	for( int i = 1 ; i < N-1 ; i++ )
		for( int j = 1 ; j < M-1 ; j++ )
			B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1]+2*A[i][j])/6.0 ;

	for( int i = 1 ; i < N-1 ; i++ )
		for( int j = 1 ; j < M-1 ; j++ )
			A[i][j] = (B[i-1][j]+B[i+1][j]+B[i][j-1]+B[i][j+1]+2*B[i][j])/6.0 ;
	}

float sum = 0 ;
for( int i = 0 ; i < N ; i++ )
	for( int j = 0 ; j < M ; j++ )
		sum +=A[i][j];

return sum ;
}


// ***************************************************************
int main(int argc, char *argv[]) {
float t ;

printf( "init Temp\n" ) ;
initTemp() ;
printf( "Compute Temp\n" ) ;

if ( argc == 1 )
	t = computeTemp(1000) ;
else
	t = computeTemp(atoi(argv[1])) ;

printf( "Sum=%f", t ) ;
   return 0;
}
