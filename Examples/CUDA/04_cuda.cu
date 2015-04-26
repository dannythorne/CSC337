
#include <iostream>
using namespace std;

__global__ void kernel( int* b, int* t)
{
  b[blockIdx.x] = blockIdx.x; // Blocks in the grid
  *t = blockDim.x; // Treads per block
}

int main()
{
  int* b;
  int* d_b;

  int t;
  int* d_t;

  int numblocks = 4;

  b = new int[numblocks];

  // store in d_b the address of a memory
  // location on the device
  cudaMalloc( (void**)&d_b, numblocks*sizeof(int));
  cudaMalloc( (void**)&d_t, sizeof(int));

  kernel<<<numblocks,1>>>(d_b,d_t);

  cudaMemcpy( b, d_b, numblocks*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( &t, d_t, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_b);
  cudaFree(d_t);

  int block;
  for( block=0; block<numblocks; block++)
  {
    cout << "blockIdx " << b[block]
         << ": " << t
         << " threads per block"
         << endl;
  }

  delete [] b;

  return 0;
}
