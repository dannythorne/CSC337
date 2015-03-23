
#include <iostream>
#include <mpi.h>
using namespace std;

int neighbor_sum( char** u2d
                , int i
                , int ip
                , int im
                , int ibyte
                , int ibytep
                , int ibytem
                , int j
                , int jp
                , int jm
                );

void update( char** u2d
           , char** u2d_next
           , int ibyte
           , int i
           , int j
           , int sum
           );

void display( char** u2d
            , int nibytes
            , int nj
            , int ibyte0
            , int j0
            , int olap
            );

void display_with_overlap(
              char** u2d
            , int nnibytes
            , int nnj
            , int myID
            );

int main( int argc, char** argv)
{

  int numProcs;
  int myID;
  MPI_Status status;

  MPI_Init( &argc, &argv);
  MPI_Comm_size( MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myID);

  cout << "Proc " << myID << " of [0," << numProcs << ")." << endl;

  int ni, nj;
  int nibytes;

  ni = 8;
  nj = 8;
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
    u2d[j] = u1d + j*nnibytes;
    u2d_next[j] = u1d_next + j*nnibytes;
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
  if( !myID)
  {
    u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<6;
    u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<5;
    u2d[j0+3][ibyte0+/*ibyte*/0] |= 1<<4;
    u2d[j0+2][ibyte0+/*ibyte*/0] |= 1<<4;
    u2d[j0+1][ibyte0+/*ibyte*/0] |= 1<<5;
  }

  cout << endl
       << "Proc " << myID << " "
       << "Universe seed:" << endl;
  display_with_overlap( u2d, nnibytes, nnj, myID);

  int t, nt = 1;
  int ibytep, ibytem;
  int ip, im;
  int jp, jm;
  int sum;

  for( t=0; t<nt; t++)
  {

    // Communicate
    if( !myID)
    {
      // Proc 0 to Proc 1
      MPI_Send(
            /* const void *buf       */ u1d + nnibytes*(nnj-2*j0)
          , /* int count             */ nnibytes*j0
          , /* MPI_Datatype datatype */ MPI_CHAR
          , /* int dest              */ 1
          , /* int tag               */ 0
          , /* MPI_Comm comm         */ MPI_COMM_WORLD
          );
    }
    else
    {
      // Proc 1 to Proc 0
      MPI_Recv(
            /* const void *buf       */ u1d + 0
          , /* int count             */ nnibytes*j0
          , /* MPI_Datatype datatype */ MPI_CHAR
          , /* int dest              */ 0
          , /* int tag               */ 0
          , /* MPI_Comm comm         */ MPI_COMM_WORLD
          , /* MPI_Status* status    */ &status
          );
    }

    cout << endl
         << "Proc " << myID << " "
         << "After communication:"
         << endl;
    display_with_overlap( u2d, nnibytes, nnj, myID);

    for( j=j0; j<j0+nj; j++)
    {
      jp = ( j-j0 + 1) % nj + j0;
      jm = ( j-j0 + nj - 1) % nj + j0;

      for( ibyte=ibyte0; ibyte<ibyte0+nibytes; ibyte++)
      {
        // msb (i=7)
        i = 7;
        ip = i - 1; // bit positions indexed backwards
        im = 0;
        ibytem = ( ibyte-ibyte0 + nibytes - 1) % nibytes + ibyte0;
        ibytep = ibyte;
        sum = neighbor_sum( u2d, i, ip, im, ibyte, ibytep, ibytem, j, jp, jm);
        update( u2d, u2d_next, ibyte, i, j, sum);

        for( i=6; i>=1; i--)
        {
          // interior bits
          ip = i - 1; // bit positions indexed backwards
          im = i + 1; // bit positions indexed backwards
          ibytem = ibyte;
          sum = neighbor_sum( u2d, i, ip, im, ibyte, ibytep, ibytem, j, jp, jm);
          update( u2d, u2d_next, ibyte, i, j, sum);
        }

        // lsb (i=0)
        i = 0;
        ip = 7;
        im = i+1; // bit positions indexed backwards
        ibytep = ( ibyte-ibyte0 + 1) % nibytes + ibyte0;
        sum = neighbor_sum( u2d, i, ip, im, ibyte, ibytep, ibytem, j, jp, jm);
        update( u2d, u2d_next, ibyte, i, j, sum);
      }
    }

    // pointer swap
    utemp = u2d;
    u2d = u2d_next;
    u2d_next = utemp;

  } // for( t=0; t<nt; t++)

  cout << endl
       << "Proc " << myID << " "
       << "Universe after " << nt << " update(s):"
       << endl;
  display_with_overlap( u2d, nnibytes, nnj, myID);

  delete [] u2d_next;
  delete [] u1d_next;
  delete [] u2d;
  delete [] u1d;

  MPI_Finalize();

  return 0;
}

int neighbor_sum( char** u2d
                , int i
                , int ip
                , int im
                , int ibyte
                , int ibytep
                , int ibytem
                , int j
                , int jp
                , int jm
                )
{
  int sum = 0;
  sum += 1&(u2d[j ][ibytem]>>im);
  sum += 1&(u2d[j ][ibytep]>>ip);
  sum += 1&(u2d[jm][ibytem]>>im);
  sum += 1&(u2d[jm][ibytep]>>ip);
  // TODO: bail when sum exceeds 3?
  sum += 1&(u2d[jp][ibytem]>>im);
  sum += 1&(u2d[jp][ibytep]>>ip);
  sum += 1&(u2d[jm][ibyte ]>>i );
  sum += 1&(u2d[jp][ibyte ]>>i );

  return sum;
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

void display_with_overlap(
              char** u2d
            , int nnibytes
            , int nnj
            , int myID
            )
{
  int i, j, ibyte;

  for( j=0; j<nnj; j++)
  {
    cout << "Proc " << myID << " ";
    for( ibyte=0; ibyte<nnibytes; ibyte++)
    {
      for( i=7; i>=0; i--)
      {
        cout << " " << (1&(u2d[j][ibyte]>>i));
      }
    }
    cout << endl;
  }
}
