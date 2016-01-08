#include <iostream>
#include <math.h>
#include <stdio.h>
#include <png.h>
#include <cuda.h>
#include <cstdio>
#include "bitmap/bitmap_image.hpp"


using namespace std;

#define DIM 1000

struct cucomplex
{
 float r, i;
 __device__ cucomplex (float a, float b) : r(a), i(b) {}
 __device__ float mag2(void) 
 {
  return r*r + i*i;
 }
 __device__ cucomplex operator*(const cucomplex& a)
 {
  return cucomplex(r*a.r - i*a.i, i*a.r + r*a.i);
 }
 __device__ cucomplex operator+(const cucomplex& a)
 {
  return cucomplex(r+a.r,i+a.i);
 }
};


__device__ int julia(int x, int y)
{
 const float scale = 1.5;
 float jx = scale*(float)(DIM/2-x)/(DIM/2);
 float jy = scale*(float)(DIM/2-y)/(DIM/2);
 
 cucomplex c(-0.8, 0.156);
 cucomplex a(jx, jy);
 
 int i = 0;
 for (i=0; i<200; i++)
 {
  a = a*a + c;
  if (a.mag2() > 1000)
    return 0;
 }
 return 1;
}


__global__ void kernel(unsigned char *ptr)
{
 int x = blockIdx.x;
 int y = blockIdx.y;
 int offset = x + y*gridDim.x;

 int juliavalue = julia(x, y);
 ptr[offset*4 + 0] = 255*juliavalue;
 ptr[offset*4 + 1] = 0;
 ptr[offset*4 + 2] = 0;
 ptr[offset*4 + 3] = 255;
}


int main( void)
{
  bitmap_image image(DIM,DIM);
  cout<<"hello-4 \n";
  unsigned char* h_i;
  unsigned char *dev_bitmap;
  cout<<"hello-3 \n";
  cudaMalloc((void**)&dev_bitmap, sizeof(int)*DIM*DIM); 
  cout<<"hello-2 \n";
  dim3 grid(DIM, DIM); 
  cout<<"hello-1 \n";
  kernel<<<grid,1>>>(dev_bitmap);
  cout<<"hello0 \n";
  h_i = (char*) malloc(DIM*DIM); 
  cudaMemcpy(h_i, dev_bitmap, sizeof(int)*DIM*DIM, cudaMemcpyDeviceToHost);
  cout<<"hello1 \n";
  image = *h_i;
  cout<<"hello2 \n"; 
  cudaFree(dev_bitmap);
  image.save_image("test.bmp");
  return 0;  
}
