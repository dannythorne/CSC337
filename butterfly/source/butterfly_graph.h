#ifndef BUTTERFLY_GRAPH_H
#define BUTTERFLY_GRAPH_H

#include <iostream>
#include <cstdlib>
#include <cmath>

struct Node{

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

    Butterfly_Graph();
    Butterfly_Graph(int n);
    ~Butterfly_Graph();

  private:

    Node** nodes;

};


#endif
