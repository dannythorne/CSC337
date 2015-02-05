#ifndef BUTTERFLY_GRAPH_H
#define BUTTERFLY_GRAPH_H

#include <iostream>
#include <cstdlib>
#include <cmath>

struct Node{

  int i;
  int j;
  Node* down;
  Node* up;
  Node* diagonal;
  bool is_left_diagonal;

};

struct Message{

  int nodes_visited;
  std::string message;

};

class Butterfly_Graph{

  public:

    Butterfly_Graph(int n);
    Butterfly_Graph();
    ~Butterfly_Graph();

    void send_message(int sender, int receiver); // void for now

  private:

    Node** nodes;
    int num_procs;
    int log_2_of_n;

};


#endif
