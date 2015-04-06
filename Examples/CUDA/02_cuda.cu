
#include <iostream>
using namespace std;

__global__ void kernel( int* n) { *n = 3;}

int main()
{
  int n;
  int* d_n;

  // store in d_n the address of a memory
  // location on the device
  cudaMalloc( (void**)&d_n, sizeof(int));

  kernel<<<1,1>>>(d_n);

  cudaMemcpy( &n, d_n, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_n);

  cout << "Hello, CUDA! " << n << endl;

  return 0;
}
