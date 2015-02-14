#ifndef BUTTERFLY_GRAPH_H
#define BUTTERFLY_GRAPH_H

#include <iostream>
#include <cstdlib>
#include <cmath>

#define MESSAGE0 "An implementation of the 2D Butterfly Topology.\n"
#define MESSAGE1 "Enter number of processors n (power of 2 except 2 or 4): "
#define MESSAGE2 "\nEnter message source (between 0 and n - 1): "
#define MESSAGE3 "\nEnter message destination (between 0 and n - 1): "

#define NUM_PROCS(x) "\nNumber of processors: " << x
#define SOURCE(y) "\nSource: " << y
#define DESTINATION(z) "\nDestination: " << z << endl

#define THREAD_CREATE_FAILURE 1
#define THREAD_JOIN_FAILURE 2

struct Node{

  int i;
  int j;
  Node* down;
  Node* up;
  Node* diagonal;
  bool is_left_diagonal;

};

class Butterfly_Graph{

  public:

    Butterfly_Graph(int n);
    Butterfly_Graph();
    ~Butterfly_Graph();
    void send_message(int sender, int receiver);

  private:

    Node** nodes;
    int num_procs;
    int log_2_of_n;
    char* input_to_binary(int input);

};

#endif
