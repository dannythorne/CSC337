#include "butterfly_graph.h"
using namespace std;

int main( int argc, char** argv )
{
  Butterfly_Graph* graph = new Butterfly_Graph(8);
  graph->send_message( 0, 6);
  return 0;
}
