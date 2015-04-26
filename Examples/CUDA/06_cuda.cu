
#include <iostream>
using namespace std;

__global__ void kernel( int* b, int* t)
{
  if( !threadIdx.x)
  {
    b[blockIdx.x] = blockIdx.x;
  }
  t[blockDim.x     *blockIdx.x + threadIdx.x] = threadIdx.x;
}

int main()
{
  int numBlocks = 4;
  int threadsPerBlock = 8;
  int numThreads = numBlocks*threadsPerBlock;

  int* b;
  int* t;

  // allocate b and t locally on host
  b = new int[numBlocks];
  t = new int[numThreads];

  int* d_b;
  int* d_t;

  // allocate d_b and d_t on the device
  cudaMalloc( (void**)&d_b, numBlocks*sizeof(int));
  cudaMalloc( (void**)&d_t, numThreads*sizeof(int));

  kernel<<<numBlocks,threadsPerBlock>>>(d_b,d_t);

  // communicate values from device to host
  cudaMemcpy( b, d_b, numBlocks*sizeof(int)
            , cudaMemcpyDeviceToHost);
  cudaMemcpy( t, d_t, numThreads*sizeof(int)
            , cudaMemcpyDeviceToHost);

  // free the memory allocated on the device
  cudaFree(d_b);
  cudaFree(d_t);

  // output results
  int block;
  int thread;
  for( block=0; block<numBlocks; block++)
  {
    cout << "b[" << block << "] = " << b[block];
    cout << "; threads:";
    for( thread=0; thread<threadsPerBlock; thread++)
    {
      cout << " " << t[block*threadsPerBlock + thread];
    }
    cout << endl;
  }

  // delete the locally allocate memory
  delete [] t;
  delete [] b;

  return 0;
}
