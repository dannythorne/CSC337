#include "butterfly_graph.h"
using namespace std;

Butterfly_Graph::Butterfly_Graph(int n)
{
  int log_2_of_n = (int)log2((double)n);

  nodes = new Node*[n];

  for(int i = 0; i < n; i++)
  {
    nodes[i] = new Node[log_2_of_n + 1];
  }

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; i < log_2_of_n; j++)
    {
      if(j == log_2_of_n)
      {
        nodes[i][j].down = NULL;
        nodes[i][j].diagonal = NULL;
        // This node is on the bottom.
      }
      else
      {
        nodes[i][j].down = &nodes[i][j + 1];
        if((i % (n / (int)pow(2, j))) <= (n / (int)pow(2, j+1)))
        {
          nodes[i][j].diagonal = &nodes[i + (n / (int)pow(2, j+1))][j - 1];
          nodes[i][j].is_left_diagonal = false;
          // This Node has a right diagonal.
        }
        else
        {
          nodes[i][j].diagonal = &nodes[i - (n / (int)pow(2, j+1))][j - 1];
          nodes[i][j].is_left_diagonal = true;
          // This Node has a left diagonal.
        }
      }
      if(j == 0)
      {
        nodes[i][j].up = NULL;
        // This node is at the top.
      }
      else
      {
        nodes[i][j].down = &nodes[i][j - 1];
      }
    }
  }
}

Butterfly_Graph::~Butterfly_Graph()
{
}

Butterfly_Graph::Butterfly_Graph()
{
}
