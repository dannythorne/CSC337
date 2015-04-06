
#include <iostream>
using namespace std;

__global__ void kernel( int* b, int* t)
{
  *b = gridDim.x; // Blocks in the grid
  *t = blockDim.x; // Treads per block
}

int main()
{
  int b;
  int* d_b;
  int t;
  int* d_t;

  // store in d_b the address of a memory
  // location on the device
  cudaMalloc( (void**)&d_b, sizeof(int));
  cudaMalloc( (void**)&d_t, sizeof(int));

  kernel<<<1,1>>>(d_b,d_t);

  cudaMemcpy( &b, d_b, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( &t, d_t, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_b);
  cudaFree(d_t);

  cout << "Num blocks           : " << b << endl;
  cout << "Num threads per block: " << t << endl;

  return 0;
}
