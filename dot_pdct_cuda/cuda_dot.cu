#include <iostream>
#include <stdio.h>

using namespace std;


#define imin(a,b) (a<b?a:b)

const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);


// dot on the kernel
__global__ void dot(float *a, float *b, float *c)
{
 __shared__ float cache[threadsPerBlock];
 int cacheIndex = threadIdx.x;

 float temp = 0.0;
 for (int tid = threadIdx.x + blockIdx.x*blockDim.x; tid<N; tid += blockDim.x*gridDim.x) 
 {
  temp += a[tid]*b[tid]; 
 }
 
 cache[cacheIndex] = temp;

 __syncthreads();


 // reduction
 for (int i = blockDim.x/2; i>0; i /= 2)
 {
  if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
  __syncthreads();
 }

 if (threadIdx.x == 0) c[blockIdx.x] = cache[0];
}


// main fn
int main(void)
{
 float *a, *b, c, *partial_c;
 float *dev_a, *dev_b, *dev_partial_c;

 a = (float*)malloc(N*sizeof(float));
 b = (float*)malloc(N*sizeof(float)); 
 partial_c = (float*)malloc(blocksPerGrid*sizeof(float)); 

 cudaMalloc((void**)&dev_a, N*sizeof(float));
 cudaMalloc((void**)&dev_b, N*sizeof(float));
 cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

 for (int i=0; i<N; i++)
 {
  a[i] = i;
  b[i] = 2*i;
 }

 cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
 dot<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
 
 cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

 c = 0.0;
 for (int i=0; i<blocksPerGrid; i++)
 {
  c += partial_c[i];
 }  

 #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
 cout<< "GPU value = "<<c<<" analytical value = "<<2*sum_squares((float)(N-1))<<endl;
 
 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(partial_c);
 
 free(a);
 free(b);
 free(partial_c);

}
