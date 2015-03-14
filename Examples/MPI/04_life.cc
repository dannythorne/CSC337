
#include <iostream>
using namespace std;

void update( char** u2d
           , char** u2d_next
           , int ibyte
           , int i
           , int j
           , int sum );

void display( char** u2d
            , int nibytes
            , int nj
            , int ibyte0
            , int j0
            , int olap );

int main()
{
  int ni, nj;
  int nibytes;

  ni = 16;
  nj = 16;
  if( ni%8) { ni = ni + 8 - ni%8;}
  nibytes = ni/8;

  int olap = 8;
  if( olap%8) { olap = olap + 8 - olap%8;} // enforce 8|olap

  int n = nibytes*nj;
  int nnj = nj + 2*olap;
  int nnibytes = nibytes + 2*olap/8;
  int nn = nnibytes*nnj;
  int ibyte0 = olap/8;
  int j0 = olap;

  char* u1d;
  char** u2d;
  char* u1d_next;
  char** u2d_next;

  char** utemp; // for pointer swap after each update

  u1d = new char[nn];
  u2d = new char*[nnj];
  u1d_next = new char[nn];
  u2d_next = new char*[nnj];

  int i, ibyte, j;

  for( j=0; j<nnj; j++)
  {
    u2d[j] = u1d + j*nibytes; // TODO
    u2d_next[j] = u1d_next + j*nibytes; // TODO
  }

  for( j=0; j<nnj; j++)
  {
    for( ibyte=0; ibyte<nnibytes; ibyte++)
    {
      u2d[j][ibyte] = 0; // j*nibytes + ibyte;
      u2d_next[j][ibyte] = 0;
    }
  }

  // Initialize with glider pattern in top left corner.
  u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<6;
  u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<5;
  u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<4;
  u2d[j0+2][ibyte0+/*ibyte*/0] |= 1<<4;
  u2d[j0+1][ibyte0+/*ibyte*/0] |= 1<<5;

  cout << endl << "Universe seed:" << endl;
  display( u2d, nibytes, nj, ibyte0, j0, olap);

  int t, nt = 64;
  int ibytep, ibytem;
  int ip, im;
  int jp, jm;
  int sum;

  for( t=0; t<nt; t++)
  {
    for( j=j0; j<j0+nj; j++)
    {
      jp = ( j-j0 + 1) % nj + j0;
      jm = ( j-j0 + nj - 1) % nj + j0;

      for( ibyte=ibyte0; ibyte<ibyte0+nibytes; ibyte++)
      {
        ibytep = ( ibyte-ibyte0 + 1) % nibytes + ibyte0;
        ibytem = ( ibyte-ibyte0 + nibytes - 1) % nibytes + ibyte0;

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
  display( u2d, nibytes, nj, ibyte0, j0, olap);

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

void display( char** u2d
            , int nibytes
            , int nj
            , int ibyte0
            , int j0
            , int olap )
{
  int i, j, ibyte;

  for( j=j0; j<j0+nj; j++)
  {
    for( ibyte=ibyte0; ibyte<ibyte0+nibytes; ibyte++)
    {
      for( i=7; i>=0; i--)
      {
        cout << " " << (1&(u2d[j][ibyte]>>i));
      }
    }
    cout << endl;
  }
}
