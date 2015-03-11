
#include <iostream>
using namespace std;

void update( char** u2d
           , char** u2d_next
           , int ibyte
           , int i
           , int j
           , int sum );

void display( char** u2d, int nibytes, int nj);

int main()
{
  int ni, nj;
  int nibytes;

  ni = 16;
  nj = 16;
  if( ni%8) { ni = ni + 8 - ni%8;}
  nibytes = ni/8;

  int n = nibytes*nj;

  char* u1d;
  char** u2d;
  char* u1d_next;
  char** u2d_next;

  char** utemp; // for pointer swap after each update

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
      u2d[j][ibyte] = 0; // j*nibytes + ibyte;
      u2d_next[j][ibyte] = 0;
    }
  }

  u2d[3][/*ibyte*/0] |= 1<<6;
  u2d[3][/*ibyte*/0] |= 1<<5;
  u2d[3][/*ibyte*/0] |= 1<<4;
  u2d[2][/*ibyte*/0] |= 1<<4;
  u2d[1][/*ibyte*/0] |= 1<<5;

  cout << endl << "Universe seed:" << endl;
  display( u2d, nibytes, nj);

  int t, nt = 64;
  int ibytep, ibytem;
  int ip, im;
  int jp, jm;
  int sum;

  for( t=0; t<nt; t++)
  {
    for( j=0; j<nj; j++)
    {
      jp = ( j + 1) % nj;
      jm = ( j + nj - 1) % nj;

      for( ibyte=0; ibyte<nibytes; ibyte++)
      {
        ibytep = ( ibyte + 1) % nibytes;
        ibytem = ( ibyte + nibytes - 1) % nibytes;

        // msb (i=7) separately
        i = 7;
        ip = i - 1; // bit positions indexed backwards
        im = 0;
        sum = 0;
        sum += 1&(u2d[j ][ibytem]>>im); // lsb of ibytem
        sum += 1&(u2d[j ][ibyte ]>>ip);
        sum += 1&(u2d[jm][ibytem]>>im); // lsb of ibytem
        sum += 1&(u2d[jm][ibyte ]>>ip);
        sum += 1&(u2d[jp][ibytem]>>im); // lsb of ibytem
        sum += 1&(u2d[jp][ibyte ]>>ip);
        sum += 1&(u2d[jm][ibyte ]>>i );
        sum += 1&(u2d[jp][ibyte ]>>i );

        update( u2d, u2d_next, ibyte, i, j, sum);

        for( i=6; i>=1; i--)
        {
          // interior bits
          ip = i - 1; // bit positions indexed backwards
          im = i + 1; // bit positions indexed backwards

          // update
          sum = 0;
          sum += 1&(u2d[j ][ibyte]>>im);
          sum += 1&(u2d[j ][ibyte]>>ip);
          sum += 1&(u2d[jm][ibyte]>>im);
          sum += 1&(u2d[jm][ibyte]>>ip);
          sum += 1&(u2d[jp][ibyte]>>im);
          sum += 1&(u2d[jp][ibyte]>>ip);
          sum += 1&(u2d[jm][ibyte]>>i );
          sum += 1&(u2d[jp][ibyte]>>i );

          update( u2d, u2d_next, ibyte, i, j, sum);
        }

        // lsb (i=0) separately
        i = 0;
        ip = 7;
        im = i+1; // bit positions indexed backwards
        sum = 0;
        sum += 1&(u2d[j ][ibyte ]>>im);
        sum += 1&(u2d[j ][ibytep]>>ip); // msb of ibytep
        sum += 1&(u2d[jm][ibyte ]>>im);
        sum += 1&(u2d[jm][ibytep]>>ip); // msb of ibytep
        sum += 1&(u2d[jp][ibyte ]>>im);
        sum += 1&(u2d[jp][ibytep]>>ip); // msb of ibytep
        sum += 1&(u2d[jm][ibyte ]>>i );
        sum += 1&(u2d[jp][ibyte ]>>i );

        update( u2d, u2d_next, ibyte, i, j, sum);
      }
    }

    // pointer swap
    utemp = u2d;
    u2d = u2d_next;
    u2d_next = utemp;

  } // for( t=0; t<nt; t++)

  cout << endl << "Universe after " << nt << " update(s):" << endl;
  display( u2d, nibytes, nj);

  delete [] u2d_next;
  delete [] u1d_next;
  delete [] u2d;
  delete [] u1d;

  return 0;
}

void update( char** u2d
           , char** u2d_next
           , int ibyte
           , int i
           , int j
           , int sum )
{
  if( 1&(u2d[j][ibyte]>>i))
  {
    // sum of 2 or 3 means leave it
    // sum of less than 2 means it dies of isolation
    // sum of > 3 means it dies of over crowding
    if( sum < 2 || sum > 3)
    {
      u2d_next[j][ibyte] &= (~(1<<i));
    }
    else
    {
      // copy the 1
      u2d_next[j][ibyte] |= (1<<i);
    }
  }
  else
  {
    // sum of 3 means the cell gives birth
    if( sum==3)
    {
      u2d_next[j][ibyte] |= (1<<i);
    }
    else
    {
      // copy the 0
      u2d_next[j][ibyte] &= (~(1<<i));
    }
  }
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
