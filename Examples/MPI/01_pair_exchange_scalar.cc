
#include <mpi.h>
#include <iostream>
using namespace std;

int main( int argc, char** argv)
{
  MPI_Init( &argc, &argv);

  int numProcs;
  int myID;
  MPI_Status status;

  int a0 = -1;
  int a1 = -1;

  MPI_Comm_size( MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myID);

  cout << "Proc " << myID << " of [0," << numProcs << ") -- " << endl;

  if( !myID)
  {
    a0 = 0; // proc 0 initializes a0
  }
  else
  {
    a1 = 1; // proc 1 initializes a1
  }

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "Before communication:  a0 = " << a0 << ", a1 = " << a1
       << endl;

  // Communicate message a0 from proc 0 to proc 1.
  if( !myID)
  {
    // proc 0 sends a0 to proc 1
    MPI_Send( &a0, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  }
  else
  {
    // proc 1 receives a0 from proc 0
    MPI_Recv( &a0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }

  // TODO: Communicate message a1 from proc 1 to proc 0.

  cout << "Proc " << myID << " of [0," << numProcs << ") -- "
       << "After  communication:  a0 = " << a0 << ", a1 = " << a1
       << endl;

  MPI_Finalize();

  return 0;
}
