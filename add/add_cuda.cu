#include <iostream>
#include <cuda.h>
#include <stdio.h>

using namespace std;

__global__ void addition(int *a, int *b, int *c)
{
   *c = *a + *b;
}


int main()
{
  int a, b, c;
  int *dev_a, *dev_b, *dev_c;
  int size = sizeof(int);

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

  cout<<"enter value for a: \n";
  cin>>a;
  cout<<"enter value for b: \n";
  cin>>b;
  
  cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);  
  cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);  

  addition<<<1,1>>>(dev_a, dev_b, dev_c);
  cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

   cudaFree(&dev_a);
   cudaFree(&dev_b);
   cudaFree(&dev_c);

   cout<<"sum of 2 numbers is: "<<c<<"\n";
   
   return 0;
}
