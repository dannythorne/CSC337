
#include <mpi.h>
#include <iostream>
using namespace std;

int main( int argc, char** argv)
{
  MPI_Init( &argc, &argv);

  int numProcs;
  int myID;
  MPI_Status status;

  double* u;
  double* unext;
  double* utemp;
  int g_n = 16;
  int n = g_n / 2; // assume g_n is even and also that n is even for now

  MPI_Comm_size( MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myID);

  cout << "Proc " << myID << " of [0," << numProcs << ") -- " << endl;

  int i;

  if( !myID)
  {
    // proc 0 initializes its portion of the global array
    //
    // +---+---+---+---+---+---+---+---+...+
    // | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 .
    // +---+---+---+---+---+---+---+---+...+
    //
    u = new double[n+1];
    unext = new double[n+1];
    u[0] = 0;
    unext[0] = 0;
    for( i=1; i<n+1; i++)
    {
      u[i] = i%2;
    }
  }
  else
  {
    // proc 1 initializes its portion of the global array
    //
    // +...+---+---+---+---+---+---+---+---+
    // . 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
    // +...+---+---+---+---+---+---+---+---+
    //
    u = new double[n+1];
    unext = new double[n+1];
    u[n] = 0;
    unext[n] = 0;
    for( i=1; i<n+1; i++)
    {
      u[i-1] = i%2;
    }
  }

  //
  //   +---+---+---+---+---+---+---+---+...+
  //   | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 .
  //   +---+---+---+---+---+---+---+---+...+
  //                               +...+---+---+---+---+---+---+---+---+
  //                               . 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
  //                               +...+---+---+---+---+---+---+---+---+
  //
  //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //   | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
  //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "After initialization:  ";

  for( i=0; i<n+1; i++)
  {
    cout << " " << u[i];
  }
  cout << endl;

  int t;
  int nt = 8;

  for( t=0; t<nt; t++)
  {
    for( i=1; i<n; i++)
    {
      unext[i] = ( u[i-1] + 2*u[i] + u[i+1]) / 4.0;
    }
    utemp = u;
    u = unext;
    unext = utemp;

    // +---+---+---+---+---+---+---+---+...+
    // |   |   |   |   |   |   |   | x |   .
    // +---+---+---+---+---+---+---+---+...+
    //                               |
    //                               |
    //                               v
    //                             +...+---+---+---+---+---+---+---+---+
    //                             . x |   |   |   |   |   |   |   |   |
    //                             +...+---+---+---+---+---+---+---+---+
    if( !myID)
    {
      // Send u[n-1] to proc 1.
      MPI_Send( /*const void *buf       */ u + (n-1)
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int dest              */ 1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              );
    }
    else
    {
      // Recv u[0] from proc 0.
      MPI_Recv( /*void *buf             */ u
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int source            */ 0
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              , /*MPI_Status *status    */ &status
              );
    }

    // +---+---+---+---+---+---+---+---+...+
    // |   |   |   |   |   |   |   |   | y .
    // +---+---+---+---+---+---+---+---+...+
    //                                   ^
    //                                   |
    //                                   |
    //                             +...+---+---+---+---+---+---+---+---+
    //                             .   | y |   |   |   |   |   |   |   |
    //                             +...+---+---+---+---+---+---+---+---+
    if( !myID)
    {
      // Recv u[n] from proc 1.
      MPI_Recv( /*void *buf             */ u + n
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int source            */ 1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              , /*MPI_Status *status    */ &status
              );
    }
    else
    {
      // Send u[1] to proc 0.
      MPI_Send( /*const void *buf       */ u + 1
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int dest              */ 0
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              );
    }

  }

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "After communication:  ";

  for( i=0; i<n+1; i++)
  {
    cout << " " << u[i];
  }
  cout << endl;

  MPI_Finalize();

  return 0;
}
