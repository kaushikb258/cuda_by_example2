#include <iostream>
#include <cuda.h>
#include <stdio.h>

using namespace std;

#define N 20


__global__ void addition(int *a, int *b, int *c)
{
 int tid = blockIdx.x;
 if (tid < N) 
   c[tid] = a[tid] + b[tid];
}


int main()
{
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;
  int size = N*sizeof(int);
  int i;

  cudaError_t err;

  err = cudaMalloc((void**)&dev_a, size);
  if(err != cudaSuccess){
   cout<<"Error1 \n";
  }
  err = cudaMalloc((void**)&dev_b, size);
  if(err != cudaSuccess){
   cout<<"Error2 \n";
  }
  err = cudaMalloc((void**)&dev_c, size);
  if(err != cudaSuccess){
   cout<<"Error3 \n";
  }

  for (i=0; i<N; i++){
   a[i] = -i;
   b[i] = i*i;
  }
  
  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);  
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);  

  addition<<<N,1>>>(dev_a, dev_b, dev_c);
  cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  for (i=0; i<N; i++){
   cout<<a[i]<<" + "<<b[i]<<" = "<<c[i]<<"\n";
  }

   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFree(dev_c);
   
   return 0;
}
