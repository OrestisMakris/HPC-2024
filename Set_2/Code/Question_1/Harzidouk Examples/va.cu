#include <iostream>
#include <cuda.h>
using namespace std;

int *a, *b;  // host data
int *c, *c2;  // results

__global__ void vecAdd(int *A, int *B, int *C, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   //int i = blockIdx.x;
   //if (i == 0)
   //    printf("i=0\n");
   C[i] = A[i] + B[i];
}

void vecAdd_h(int *A1,int *B1, int *C1, int N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] + B1[i];
}

int main(int argc,char **argv)
{
   printf("Begin \n");
   int n=64;
   int nBytes = n*sizeof(int);

   a = (int *)malloc(nBytes);
   b = (int *)malloc(nBytes);
   c = (int *)malloc(nBytes);
   c2 = (int *)malloc(nBytes);

   int *a_d,*b_d,*c_d;

   for(int i=0;i<n;i++)
   {
      a[i]=i;
      b[i]=i;
   }


   printf("Allocating device memory on host..\n");
   cudaMalloc((void **)&a_d,n*sizeof(int));
   cudaMalloc((void **)&b_d,n*sizeof(int));
   cudaMalloc((void **)&c_d,n*sizeof(int));

   printf("Copying to device..\n");
   cudaMemcpy(a_d, a, n*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b, n*sizeof(int),cudaMemcpyHostToDevice);

   printf("Doing GPU Vector add\n");
   //vecAdd<<<block_no,block_size>>>(a_d,b_d,c_d,n);
   vecAdd<<<n/4,4>>>(a_d,b_d,c_d,n);
   cudaMemcpy(c,c_d,n*sizeof(int),cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   clock_t end_d = clock();

   for(int i=0; i<n;i++)
       printf("c[%d]->%d\n", i, c[i]);

   printf("Doing CPU Vector add\n");
   vecAdd_h(a,b,c2,n);

   //for(int i=0; i<n;i++)
   //    printf("c2[%d]->%d\n", i, c2[i]);


   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);
   return 0;
}