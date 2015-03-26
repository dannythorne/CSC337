#include "butterfly_graph.h"
#include "pthread.h"
#include "msg_def.h"
using namespace std;

void* make_graph(void* size);
int get_n();
int get_source(int n);
int get_destination(int n);

int main(int argc, char** argv)
{
  int n, source, destination;
  void* thread_graph;
  pthread_t init_thread;

  n = get_n();
  if(pthread_create(&init_thread, NULL, &make_graph, &n) != 0) return THREAD_CREATE_FAILURE;
  source = get_source(n);
  destination = get_destination(n);
  if(pthread_join(init_thread, &thread_graph) != 0) return THREAD_JOIN_FAILURE;
  Butterfly_Graph* graph = static_cast<Butterfly_Graph*>(thread_graph);
  graph->send_message(source, destination);
  return 0;
}

void* make_graph(void* size)
{
  int* ptr = static_cast<int*>(size);
  void* graph = new Butterfly_Graph(*ptr);
  return graph;
}

int get_n()
{
  int n;
  int result;
  cout << MESSAGE0;
  do {
    cout << MESSAGE1;
    cin >> n;
    result = (n & (n - 1));
  } while( (n <= 4) || (result != 0) );
  return n;
}

int get_source(int n)
{
  int source;
  do {
    cout << MESSAGE2;
    cin >> source;
  } while( (source > (n - 1)) || (source < 0) );
  return source;
}

int get_destination(int n)
{
  int destination;
  do {
    cout << MESSAGE3;
    cin >> destination;
  } while( (destination > (n - 1)) || (destination < 0) );
  return destination;
}
