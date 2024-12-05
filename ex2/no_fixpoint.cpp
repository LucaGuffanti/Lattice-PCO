#include <stack>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

#include "parser.hpp"


// Tree node
class Node {
public:
  int depth; // depth in the tree
  std::vector<int> assignments;

  Node(size_t N): depth(0), assignments(N) {
    std::fill(assignments.begin(), assignments.end(), 0);
  }
  Node(const Node&) = default;
  Node(Node&&) = default;
  Node() = default;
};

bool check_if_safe(
    const std::vector<int>& assignments,
    Data& d,
    const int& interacting_nodes,
    const int& possible_assignment,
    const int& current_node
)
{
    for (int i = 0; i < interacting_nodes; ++i)
    {
        if (d.get_C_at(i, current_node) == 1 && assignments[i] == possible_assignment)
            return false;
    }
    return true;
}

void evaluate_and_branch(
    const Node& parent,
    std::stack<Node>& pool,
    size_t& visited_nodes,
    size_t& num_solutions,
    const size_t& N,
    Data& d
)
{
    int depth = parent.depth;

    if (depth == N) 
    {
        // print the solution contained in parent.assignments
        // std::cout << "Solution found: ";
        // for (int i = 0; i < N; ++i)
        // {
        //     std::cout << parent.assignments[i] << " ";
        // }
        // std::cout << std::endl;
        num_solutions++;
    }
    else
    {
        for (int j = 0; j <= d.get_u_at(depth); ++j)
        {
            if (check_if_safe(parent.assignments, d, depth, j, depth))
            {
                Node child(parent);
                child.assignments[depth] = j;
                child.depth++;
                pool.push(std::move(child));
                visited_nodes++;
            }
        }
    }
}


int main(int argc, char** argv)
{
    // Verify the correctness of the input parameters
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    // Read the input file and store the data
    Data d;
    d.read_input(argv[1]);

    // Initialize the number of nodes
    std::size_t N = d.get_n();
    std::cout << "Solving the problem for N = " << N << "\n";

    // Initialize the root node
    Node root(N);

    // Initialize the stack of nodes in which the only node is the root
    std::stack<Node> pool;
    pool.push(std::move(root));

    // Statistics
    std::size_t exploredNodes = 0;
    std::size_t numSolutions = 0;

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    while (!pool.empty())
    {
        Node currentNode(std::move(pool.top()));
        pool.pop();

        // check the board configuration
        evaluate_and_branch(currentNode, pool, exploredNodes, numSolutions, N, d);
    }

    // Stop the timer
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Outputs
    std::cout << "Time taken: " << duration.count() << " milliseconds\n";
    std::cout << "Number of explored nodes: " << exploredNodes << "\n";
    std::cout << "Number of solutions: " << numSolutions << "\n";

    return 0;
}