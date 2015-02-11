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

    void send_message(int sender, int receiver);

  private:

    Node** nodes;
    int num_procs;
    int log_2_of_n;

    //TODO: A private method that will take an integer make it binary with the
    //correct number of bits based on num_procs. This method is to be called in send_message.
    char* input_to_binary(int input);

};


#endif
