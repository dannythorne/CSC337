
#include <iostream>
using namespace std;

__global__ void kernel( int* b, int* t)
{
  if( !threadIdx.x)
  {
    *b = blockDim.x; // num threads per block
  }
  t[threadIdx.x] = threadIdx.x;
}

int main()
{
  int numthreads = 4;

  int b;
  int* t;

  t = new int[numthreads];

  int* d_b; // pointer to device memory
  int* d_t; // pointer to device memory

  cudaMalloc( (void**)&d_b, sizeof(int));
  cudaMalloc( (void**)&d_t, numthreads*sizeof(int));

  kernel<<<1,numthreads>>>( d_b, d_t);

  cudaMemcpy( &b, d_b, sizeof(int)
            , cudaMemcpyDeviceToHost);

  cudaMemcpy( t, d_t, numthreads*sizeof(int)
            , cudaMemcpyDeviceToHost);

  cout << "blockDim.x = " << b << endl;

  int thread;
  for( thread=0; thread<numthreads; thread++)
  {
    cout << "thread " << thread
         << ": " << t[thread]
         << endl;
  }

  cudaFree(d_t);
  cudaFree(d_b);

  return 0;
}
