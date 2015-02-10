
#include <iostream>
#include <iomanip>
using namespace std;

// void get_u2d( double** &u2d, double* u1d, int ni, int nj);
//
// Build a logical 2d interface to the memory pointed at by u1d.
//
// PRECONDITIONS:
//   u1d points to allocated memory of length ni*nj*sizeof(double)
//   u2d is uninitialized
//
// POSTCONDITIONS
//   u2d points to an array of pointers of length nj
//   u2d[0] points to u1d
//   u2d[1] points to u1d + ni
//   u2d[2] points to u1d + ni*2
//   .
//   .
//   .
//   u2d[j] points to u1d + ni*j
//   .
//   .
//   .
//   u2d[nj-1] points to u1d + ni*(nj-1)
//
void get_u2d( double** &u2d, double* u1d, int ni, int nj);

int main()
{
  double*  u1d;
  double** u2d;

  int ni = 8;
  int nj = 6;

  int n = ni*nj;

  int i, j, k;

  u1d = new double[n];

  for( k=0; k<n; k++)
  {
    u1d[k] = k+1;
  }

  get_u2d( u2d, u1d, ni, nj);

  for( j=0; j<nj; j++)
  {
    for( i=0; i<ni; i++)
    {
      cout << " " << setw(2) << u2d[j][i];
    }
    cout << endl;
  }

  delete [] u2d;
  delete [] u1d;

  return 0;
}

void get_u2d( double** &u2d, double* u1d, int ni, int nj)
{
  // TODO
}
