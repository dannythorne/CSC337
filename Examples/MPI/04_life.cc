
#include <iostream>
using namespace std;

void display( char** u2d, int nibytes, int nj);

int main()
{
  int ni, nj;
  int nibytes;

  ni = 16;
  nj = 16;
  if( 0/* ni not a multiple of 8*/) { /* fix it */ }
  nibytes = ni/8;

  int n = nibytes*nj;

  char* u1d;
  char** u2d;
  char* u1d_next;
  char** u2d_next;

  u1d = new char[n];
  u2d = new char*[nj];
  u1d_next = new char[n];
  u2d_next = new char*[nj];

  int i, ibyte, j;

  for( j=0; j<nj; j++)
  {
    u2d[j] = u1d + j*nibytes;
    u2d_next[j] = u1d_next + j*nibytes;
  }

  for( j=0; j<nj; j++)
  {
    for( ibyte=0; ibyte<nibytes; ibyte++)
    {
      u2d[j][ibyte] = j*nibytes + ibyte;
      u2d_next[j][ibyte] = 0;
    }
  }

  cout << endl << "Universe seed:" << endl;
  display( u2d, nibytes, nj);

  int t, nt = 1;

  for( t=0; t<nt; t++)
  {
    for( j=0; j<nj; j++)
    {
      for( ibyte=0; ibyte<nibytes; ibyte++)
      {
        for( i=7; i>=0; i--)
        {
          // TODO: u2d_next[][] = f( u2d[][]);
          // where f is the GoL update rules
          // (periodic boundaries)
        }
      }
    }
    // TODO: pointer swap
  }

  cout << endl << "Universe after " << nt << " update(s):" << endl;
  display( u2d, nibytes, nj);

  delete [] u2d_next;
  delete [] u1d_next;
  delete [] u2d;
  delete [] u1d;

  return 0;
}

void display( char** u2d, int nibytes, int nj)
{
  int i, j, ibyte;

  for( j=0; j<nj; j++)
  {
    for( ibyte=0; ibyte<nibytes; ibyte++)
    {
      for( i=7; i>=0; i--)
      {
        cout << " " << (1&(u2d[j][ibyte]>>i));
      }
    }
    cout << endl;
  }
}
