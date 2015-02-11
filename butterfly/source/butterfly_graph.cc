#include "butterfly_graph.h"
using namespace std;

Butterfly_Graph::Butterfly_Graph(int n)
{
  num_procs = n;
  log_2_of_n = (int)log2((double)n);

  nodes = new Node*[n];

  for(int i = 0; i < n; i++)
  {
    nodes[i] = new Node[log_2_of_n + 1];
  }

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < log_2_of_n + 1; j++)
    {
      nodes[i][j].i = i;
      nodes[i][j].j = j;

      if(j == log_2_of_n)
      {
        nodes[i][j].down = NULL;
        nodes[i][j].diagonal = NULL;
        // This node is on the bottom.
      }
      else
      {
        nodes[i][j].down = &nodes[i][j + 1];
        if( (i % (n / (int)pow(2.0, (double)j))) < (n / (int)pow(2.0, (double)(j+1))))
        {
          nodes[i][j].diagonal = &nodes[i + (n / (int)pow(2.0, (double)(j+1)))][j + 1];
          nodes[i][j].is_left_diagonal = false;
          // This Node has a right diagonal.
        }
        else
        {
          nodes[i][j].diagonal = &nodes[i - (n / (int)pow(2.0, (double)(j+1)))][j + 1];
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
        nodes[i][j].up = &nodes[i][j - 1];
      }
    }
  }
  cout << "Exit" << endl;
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < log_2_of_n + 1; j++)
    {
      cout << "The Current Node: " << endl;
      cout << nodes[i][j].i << " ";
      cout << nodes[i][j].j << endl;
#if 0
      cout << "The Node above: " << endl;
      cout << nodes[i][j].up->i << " ";
      cout << nodes[i][j].up->j << endl;
#endif
      if(nodes[i][j].down != NULL)
      {
        cout << "The Node below: " << endl;
        cout << nodes[i][j].down->i << " ";
        cout << nodes[i][j].down->j << endl;
        cout << "Is left diagonal?: " << endl;
        cout << nodes[i][j].is_left_diagonal << endl;

        cout << "The Diagonal Node: " << endl;
        cout << nodes[i][j].diagonal->i << " ";
        cout << nodes[i][j].diagonal->j << endl;
      }
      cout << endl << endl;
    }
  }
}

Butterfly_Graph::~Butterfly_Graph()
{
}

Butterfly_Graph::Butterfly_Graph()
{
}

void Butterfly_Graph::send_message(int sender, int receiver)
{
  char* test1 = input_to_binary(sender);
  char* test2 = input_to_binary(receiver);

  cout << test1 << endl;
  cout << test2 <<endl;

#if 0
  sender = 0;
  int i = sender;

  int local_i = 0;
  int local_j = 0;

  for(int j = 0; j < log_2_of_n; j = local_j)
  {
    cout << "local i: " << local_i << endl;
    cout << "local j: " << local_j << endl;
    i = local_i;

    if( nodes[i][j].down == NULL ) {
      cout << "down is NULL" << endl;
    }

    if( nodes[i][j].diagonal == NULL ) {
      cout << "diagonal is NULL" << endl;
    }

    if(test1[j] == '1')
    {
      if(nodes[i][j].is_left_diagonal == false)
      {
        local_i = nodes[i][j].diagonal->i;
        local_j = nodes[i][j].diagonal->j;
      }
      else
      {
        local_i = nodes[i][j].down->i;
        local_j = nodes[i][j].down->j;
      }
    }
    else
    {
      if(nodes[i][j].is_left_diagonal == true)
      {
        local_i = nodes[i][j].diagonal->i;
        local_j = nodes[i][j].diagonal->j;
      }
      else
      {
        local_i = nodes[i][j].down->i;
        local_j = nodes[i][j].down->j;
      }
    }
  }
  cout << "cout << local_i << endl;" << endl;
  cout << local_i << endl;
  cout << local_j << endl;
#endif
}

char* Butterfly_Graph::input_to_binary(int input)
{
  int num_digits = log_2_of_n;
  char* result = new char[num_digits];

  for(int i = num_digits - 1; i >= 0; i--)
  {
    if(input % 2)
    {
      result[i] = '1';
    }
    else
    {
      result[i] = '0';
    }
    input /= 2;
  }

  return result;
}
