
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

  MPI_Comm_size( MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myID);

  int n = g_n / numProcs; // assume g_n is divisible by numProcs and also that numProcs is even for now

  cout << "Proc " << myID << " of [0," << numProcs << ") -- 00" << endl;

  int i;

  u = new double[n+2];
  unext = new double[n+2];
  for( i=0; i<n+2; i++)
  {
    u[i]     = myID + 1;
    unext[i] = myID + 1;
  }
  u[0] = 0.0;
  u[n+1] = 0.0;
  unext[0] = 0.0;
  unext[n+1] = 0.0;

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "01 After initialization:  ";

  for( i=0; i<n+2; i++)
  {
    cout << " " << u[i];
  }
  cout << endl;

  int t;
  int nt = 0;

  for( t=0; t<nt; t++)
  {
    // Communication step(s)
    if( 1-myID%2)
    {
      // Send in positive direction.
      MPI_Send( /*const void *buf       */ u + n
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int dest              */ myID+1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              );

      // Recv from positive direction.
      MPI_Recv( /*void *buf             */ u + (n+1)
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int source            */ myID+1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              , /*MPI_Status *status    */ &status
              );
    }
    else
    {
      // Recv from negative direction.
      MPI_Recv( /*void *buf             */ u
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int source            */ myID-1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              , /*MPI_Status *status    */ &status
              );

      // Send in negative direction.
      MPI_Send( /*const void *buf       */ u + 1
              , /*int count             */ 1
              , /*MPI_Datatype datatype */ MPI_DOUBLE
              , /*int dest              */ myID-1
              , /*int tag               */ 0
              , /*MPI_Comm comm         */ MPI_COMM_WORLD
              );
    }

    if( 1-myID%2)
    {
      // Send in negative direction.
      // TODO

      // Recv from negative direction.
      // TODO
    }
    else
    {
      // Recv from positive direction.
      // TODO

      // Send in positive direction.
      // TODO
    }

    cout << "Proc " << myID << " of [0," << numProcs << ") -- "
         << "02 After communication :  ";

    for( i=0; i<n+2; i++)
    {
      cout << " " << u[i];
    }
    cout << endl;

    for( i=1; i<n+1; i++)
    {
      unext[i] = ( u[i-1] + 2*u[i] + u[i+1]) / 4.0;
    }
    utemp = u;
    u = unext;
    unext = utemp;
  }

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "03 After updates       :  ";

  for( i=0; i<n+2; i++)
  {
    cout << " " << u[i];
  }
  cout << endl;

  double usum = 0.0;
  for( i=1; i<n+1; i++)
  {
    usum += u[i];
  }

  double g_usum;

  MPI_Reduce( /* const void *sendbuf   */ &usum
            , /* void *recvbuf         */ &g_usum
            , /* int count             */ 1
            , /* MPI_Datatype datatype */ MPI_DOUBLE
            , /* MPI_Op op             */ MPI_SUM
            , /* int root              */ 0
            , /* MPI_Comm comm         */ MPI_COMM_WORLD
            );

  if( !myID)
  {
    cout << "g_usum = " << g_usum << endl;
  }

  delete [] u;
  delete [] unext;

  MPI_Finalize();

  return 0;
}
