
#include <mpi.h>
#include <iostream>
using namespace std;

int main( int argc, char** argv)
{
  MPI_Init( &argc, &argv);

  int numProcs;
  int myID;

  MPI_Comm_size( MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &myID);

  cout << "Proc " << myID << " of [0," << numProcs << ") -- " << endl;

  MPI_Finalize();

  return 0;
}
