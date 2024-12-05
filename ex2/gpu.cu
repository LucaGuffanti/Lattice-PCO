#include <iostream>
#include <vector>
#include <chrono>
#include <stack>
#include <numeric>
#include <algorithm>
#include <bitset>
#include <cstdlib>

#include "parser.hpp"
#include <cuda_runtime.h>

int* d_constraints;
int* d_u;
size_t* d_offsets;
bool* d_changed;
bool* d_no_solution;
size_t* d_num_of_0s;

size_t N;
size_t unused_bits;
size_t total_elements;
size_t total_bytes;
size_t* offsets;
size_t* num_of_0s;
int* u;
char* singleton_domains;
char* domains;


typedef struct {
    size_t N;
    size_t total_bytes;
    
    char* domains;
    char* singleton_domains;
    char* checked_domains;
    size_t* cancelled_values;
    size_t* remaining_value;
} GPUNode;

void check_error(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::cerr << cudaGetErrorName(err) << std::endl;
        
        exit(EXIT_FAILURE);
    }
}

GPUNode* instantiate_gpu_node(const size_t& N, const size_t& total_bytes)
{
    cudaError_t err;

    // Allocate GPUNode in unified memory (accessible by both host and device)
    GPUNode* node;
    err = cudaMallocManaged(&node, sizeof(GPUNode)); check_error(err, "cudaMallocManaged-Node");

    // Initialize const members
    node->N = N;
    node->total_bytes = total_bytes;

    // Allocate member arrays on the device
    err = cudaMallocManaged(&node->domains, total_bytes); check_error(err, "cudaMalloc-Domains");
    err = cudaMallocManaged(&node->singleton_domains, (N + 7) / 8); check_error(err, "cudaMalloc-SingletonDomains");
    err = cudaMallocManaged(&node->checked_domains, (N + 7) / 8); check_error(err, "cudaMalloc-CheckedDomains");
    err = cudaMallocManaged(&node->cancelled_values, N * sizeof(size_t)); check_error(err, "cudaMalloc-cancelledValues");
    err = cudaMallocManaged(&node->remaining_value, N * sizeof(size_t)); check_error(err, "cudaMalloc-RemainingValue");

    return node;
}

void instantiate_additional_data(Data& d)
{
    // Read the data from the Data object
    N = d.get_n();
    total_elements = std::accumulate(d.get_u(), d.get_u() + N, 0) + N;
    total_bytes = (total_elements + 7) / 8;
    offsets = new size_t[N];
    offsets[0] = 0;
    for (size_t i = 1; i < N; ++i)
    {
        offsets[i] = offsets[i-1] + d.get_u_at(i-1) + 1;
    }

    // Print the data
    std::cout << "N              : " << N << std::endl;
    std::cout << "Total elements : " << total_elements << std::endl;
    std::cout << "Total bytes    : " << total_bytes << std::endl;
    std::cout << "Offsets        : ";
    for (size_t i = 0; i < N; ++i)
    {
        std::cout << offsets[i] << " ";
    }
    std::cout << std::endl;

    // Linearize the constraint matrix
    int* constraints = new int[N * N];
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            constraints[i * N + j] = d.get_C_at(i, j);
        }
    }

    // Allocate memory on the device
    cudaError_t err;
    err = cudaMalloc(&d_constraints, N * N * sizeof(int)); check_error(err, "cudaMalloc-Constraints");
    err = cudaMalloc(&d_u, N * sizeof(int)); check_error(err, "cudaMalloc-U");
    err = cudaMalloc(&d_offsets, N * sizeof(size_t)); check_error(err, "cudaMalloc-Offsets");
    err = cudaMalloc(&d_changed, sizeof(bool)); check_error(err, "cudaMalloc-Changed");
    err = cudaMalloc(&d_no_solution, sizeof(bool)); check_error(err, "cudaMalloc-NoSolution");
    err = cudaMallocManaged(&d_num_of_0s, sizeof(size_t) * N); check_error(err, "cudaMalloc-AllSingletons");
    
    // Copy the data to the device
    err = cudaMemcpy(d_constraints, constraints, N * N * sizeof(int), cudaMemcpyHostToDevice); check_error(err, "cudaMemcpy-Constraints");
    err = cudaMemcpy(d_u, d.get_u(), N * sizeof(int), cudaMemcpyHostToDevice); check_error(err, "cudaMemcpy-U");
    err = cudaMemcpy(d_offsets, offsets, N * sizeof(size_t), cudaMemcpyHostToDevice); check_error(err, "cudaMemcpy-Offsets");

} 

void free_gpu_node(GPUNode* node)
{
    cudaError_t err;

    // Free member arrays
    err = cudaFree(node->domains); check_error(err, "cudaFree-Domains");
    err = cudaFree(node->singleton_domains); check_error(err, "cudaFree-SingletonDomains");
    err = cudaFree(node->checked_domains); check_error(err, "cudaFree-CheckedDomains");
    err = cudaFree(node->cancelled_values); check_error(err, "cudaFree-cancelledValues");
    err = cudaFree(node->remaining_value); check_error(err, "cudaFree-RemainingValue");

    // Free the GPUNode structure itself
    err = cudaFree(node); check_error(err, "cudaFree-Node");
}

void free_additional_data()
{
    cudaError_t err;

    // Free the memory on the device
    err = cudaFree(d_constraints); check_error(err, "cudaFree-Constraints");
    err = cudaFree(d_u); check_error(err, "cudaFree-U");
    err = cudaFree(d_offsets); check_error(err, "cudaFree-Offsets");
    err = cudaFree(d_changed); check_error(err, "cudaFree-Changed");
    err = cudaFree(d_no_solution); check_error(err, "cudaFree-NoSolution");
    err = cudaFree(d_num_of_0s); check_error(err, "cudaFree-AllSingletons");
}

void copy_data_from_parent_to_child(GPUNode* parent, GPUNode* child)
{
    cudaError_t err;

    // Copy the data from the source to the destination
    child->N = parent->N;
    child->total_bytes = parent->total_bytes;

    err = cudaMemcpy(child->domains, parent->domains, parent->total_bytes, cudaMemcpyDeviceToDevice); check_error(err, "cudaMemcpy-DeviceToDevice-Domains");
    err = cudaMemcpy(child->singleton_domains, parent->singleton_domains, (parent->N + 7) / 8, cudaMemcpyDeviceToDevice); check_error(err, "cudaMemcpy-DeviceToDevice-SingletonDomains");
    err = cudaMemcpy(child->checked_domains, parent->checked_domains, (parent->N + 7) / 8, cudaMemcpyDeviceToDevice); check_error(err, "cudaMemcpy-DeviceToDevice-CheckedDomains");
    err = cudaMemcpy(child->cancelled_values, parent->cancelled_values, parent->N * sizeof(size_t), cudaMemcpyDeviceToDevice); check_error(err, "cudaMemcpy-DeviceToDevice-cancelledValues");
    err = cudaMemcpy(child->remaining_value, parent->remaining_value, parent->N * sizeof(size_t), cudaMemcpyDeviceToDevice); check_error(err, "cudaMemcpy-DeviceToDevice-RemainingValue");
}

__global__ void fixpoint_kernel(GPUNode* current, size_t* d_offsets, int* d_constraints, int* d_u)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current->N) return;

    // Check if a domain is a singleton
    if (current->singleton_domains[idx / 8] & (1 << (idx % 8)))
    {
        const int value = current->remaining_value[idx];
        // If the domain is a singleton, we need to apply the constraints
        for (size_t i = 0; i < current->N; ++i)
        {   
            if (idx == i) continue;

            if (d_constraints[idx * current->N + i] == 1)
            {
                // If the value is inside the domain of the other variable
                if (value <= d_u[i])
                {
                    const size_t byte_idx = (d_offsets[i] + value) / 8;
                    const size_t bit_idx = (d_offsets[i] + value) % 8;
                    if (!(current->domains[byte_idx] & (1 << bit_idx)))
                    {
                        current->domains[byte_idx] |= (1 << bit_idx);
                    }
                }
            }
        }
    }
}


__global__ void check_singleton_domains(GPUNode* current, size_t* num_of_0s)
{
    // Map a thread to each byte of the current->domains array
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= current->total_bytes) return;

    // Count the number of 0s in the byte
    char byte = current->domains[idx];
    int count = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (!(byte & (1 << i)))
        {
            count++;
        }
    }
    num_of_0s[idx] = count;
}

__global__ void count_num_of_0s(char* current, size_t total_bytes, size_t* num_of_0s)
{
    // Map a thread to each byte of the current->domains array
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) return;

    // Count the number of 0s in the byte
    char byte = current[idx];
    int count = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (!(byte & (1 << i)))
        {
            count++;
        }
    }
    num_of_0s[idx] = count;
}


void fixpoint(GPUNode* current, std::stack<GPUNode*>& stack, size_t& exploredTree, size_t& exploredSol)
{
    cudaError_t err;
    bool changed = true;
    bool no_solution = false;


    while (changed)
    {
        changed = false;
        err = cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice); check_error(err, "cudaMemcpy-HostToDevice-Changed");
        fixpoint_kernel<<<(current->N + 31) / 32, 32>>>(current, d_offsets, d_constraints, d_u);
        err = cudaDeviceSynchronize(); check_error(err, "cudaDeviceSynchronize");
        err = cudaGetLastError(); check_error(err, "fixpoint_kernel");

        count_num_of_0s<<<(current->total_bytes + 31) / 32, 32>>>(current->domains, current->total_bytes, d_num_of_0s);
        err = cudaDeviceSynchronize(); check_error(err, "cudaDeviceSynchronize");
        err = cudaGetLastError(); check_error(err, "count_num_of_0s");
        
        const size_t num_of_0s = std::accumulate(d_num_of_0s, d_num_of_0s + current->total_bytes, 0);
        if (num_of_0s == current->N + unused_bits)
        {
            exploredSol++;
            return;
        }
        if (num_of_0s < current->N + unused_bits)
        {
            return;
        }

        // Now iterate over the domains that are not singletons
        for (size_t domain = 0; domain < current->N; ++domain)
        {
            if (!(current->singleton_domains[domain / 8] & (1 << (domain % 8))))
            {
                // Count all the 0s in the domain
                size_t count = 0;
                for (size_t i = 0; i <= u[domain]; ++i)
                {
                    const size_t byte_idx = (offsets[domain] + i) / 8;
                    const size_t bit_idx = (offsets[domain] + i) % 8;
                    if (!(current->domains[byte_idx] & (1 << bit_idx)))
                    {
                        count++;
                    }
                }
                if (count == 1)
                {
                    // Set the domain as singleton
                    current->singleton_domains[domain / 8] |= (1 << (domain % 8));
                    // Find the value that is 0
                    size_t value = 0;
                    for (size_t i = 0; i <= u[domain]; ++i)
                    {
                        const size_t byte_idx = (offsets[domain] + i) / 8;
                        const size_t bit_idx = (offsets[domain] + i) % 8;
                        if (!(current->domains[byte_idx] & (1 << bit_idx)))
                        {
                            value = i;
                            break;
                        }
                    }
                    current->remaining_value[domain] = value;
                    current->cancelled_values[domain] = u[domain];
                    changed = true;
                }   
            }
        }

    }


    if (std::accumulate(d_num_of_0s, d_num_of_0s + current->total_bytes, 0) == N + unused_bits)
    {
        if(exploredSol%1000 == 0)
        {
            std::cout << "Explored tree  : " << exploredSol << std::endl;
        }
        exploredSol++;
        return;
    }

    // Else, we need to select the first non-singleton domain
    // Copy the singleton domains from the device to the host

    err = cudaMemcpy(singleton_domains, current->singleton_domains, (N+7)/8, cudaMemcpyDeviceToHost); check_error(err, "cudaMemcpy-DeviceToHost-SingletonDomains");


    // Find the first non-singleton domain
    size_t branching_variable = 0;
    for (branching_variable = 0; branching_variable < N; ++branching_variable)
    {
        if (!(singleton_domains[branching_variable / 8] & (1 << (branching_variable % 8))))
        {
            break;
        }
    }

    // Actually branch on the variable
    err = cudaMemcpy(domains, current->domains, current->total_bytes, cudaMemcpyDeviceToHost); check_error(err, "cudaMemcpy-DeviceToHost-Domains");
    for (int branch_value = 0; branch_value <= u[branching_variable]; ++branch_value)
    {
        const size_t byte_idx = (offsets[branching_variable] + branch_value) / 8;
        const size_t bit_idx = (offsets[branching_variable] + branch_value) % 8;
        // Branch if the variable has a 0
        if (!(domains[byte_idx] & (1 << bit_idx)))
        {
            exploredTree++; 
            // std::cout << "Branching on value " << branch_value << std::endl;
            GPUNode* child = instantiate_gpu_node(N, total_bytes);
            // Copy the data from the parent to the child
            copy_data_from_parent_to_child(current, child);
            // Set the domain as a singleton
            child->singleton_domains[branching_variable / 8] |= (1 << (branching_variable % 8));
            // Set the remaining value
            child->remaining_value[branching_variable] = branch_value;
            // Cancel all the other values
            for (int i = 0; i <= u[branching_variable]; ++i)
            {
                if (i != branch_value)
                {
                    const size_t byte_idx = (offsets[branching_variable] + i) / 8;
                    const size_t bit_idx = (offsets[branching_variable] + i) % 8;
                    child->domains[byte_idx] |= (1 << bit_idx);
                }
            }
            
            stack.push(child);
        }
    }
}


void test_gpu_fixpoint(char* str)
{
    Data d;
    d.read_input(str);

    instantiate_additional_data(d);

    GPUNode* root = instantiate_gpu_node(N, total_bytes);

    num_of_0s = new size_t[total_bytes];
    singleton_domains = new char[(N+7)/8];
    domains = new char[total_bytes];
    unused_bits = 8 * total_bytes - total_elements;

    u = d.get_u();
    std::stack<GPUNode*> stack;
    stack.push(root);

    size_t exploredTree = 0;
    size_t exploredSol = 0;

    std::cout << "Starting the fixpoint algorithm" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while (!stack.empty())
    {
        GPUNode* current = stack.top();
        stack.pop();

        fixpoint(current, stack, exploredTree, exploredSol);

        free_gpu_node(current);
    }

    auto end = std::chrono::high_resolution_clock::now();

    free_additional_data();
    std::cout << "===================================" << std::endl;
    std::cout << "Explored tree  : " << exploredTree << std::endl;
    std::cout << "Explored sol   : " << exploredSol << std::endl;
    std::cout << "Time           : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

}


int main(int argc, char** argv)
{  
    test_gpu_fixpoint(argv[1]);
    
    return 0;
}
