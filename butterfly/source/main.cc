#include "butterfly_graph.h"
using namespace std;

#define MESSAGE0 "An implementation of the 2D Butterfly Topology.\n"
#define MESSAGE1 "Enter number of processors n (power of 2 except 2 or 4): \n"
#define MESSAGE2 "Enter message source (between 0 and n - 1): \n"
#define MESSAGE3 "Enter message destination (between 0 and n - 1): \n"

int main( int argc, char** argv )
{
  int n;
  int source;
  int destination;
  int result;

  cout << MESSAGE0;

  do
  {
    cout << MESSAGE1;
    cin >> n;
    result = (int)(n & (n - 1));
  } while( ((n == 0) || (n == 1) || (n == 2) || (n == 4)) || (result != 0) );

  do
  {
    cout << MESSAGE2;
    cin >> source;
  } while ( source > (n - 1) || source < 0 );

  do
  {
    cout << MESSAGE3;
    cin >> destination;
  } while ( destination > (n - 1) || destination < 0 );

  cout << "Number of processors: " << n << endl;
  cout << "Source: " << source << endl;
  cout << "Destination: " << destination << endl;


  Butterfly_Graph* graph = new Butterfly_Graph(n);
  graph->send_message(source, destination);

  return 0;
}
