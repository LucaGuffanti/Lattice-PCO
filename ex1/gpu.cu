#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cassert>

#include <cuda_runtime.h>
#include <omp.h>


#define NUM_THREADS 1024

int NUM_BLOCKS;

void cudaCheckError(cudaError_t error, const char* file, const int line)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__global__ void min_fixpoint(int* arr, const size_t size, int* m)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    // Load m into my_min in an atomic way
    
    for (size_t i = tid; i < size; i = i + stride)
    {
        int my_min = *m;
        if (arr[i] < my_min) {
            // Store into m the value of arr[i] in an atomic way
            *m = arr[i];
        }
    }
    return;
}

__host__ void run_min_fixpoint(int* arr, const size_t size, int* m)
{
    int* d_arr;
    int* d_min;
    int max = INT_MAX;
    int computed_min;

    cudaError_t error;
    // Allocate memory on the device
    error = cudaMalloc(&d_arr, size * sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMalloc(&d_min, sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);

    // Copy the array to the device
    error = cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMemcpy(d_min, &max, sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);

    computed_min = max;
    int old = 0;


    while(old != computed_min)
    {
        old = computed_min;
        min_fixpoint<<<NUM_BLOCKS, NUM_THREADS>>>(d_arr, size, d_min);
        error = cudaDeviceSynchronize(); cudaCheckError(error, __FILE__, __LINE__);
        error = cudaGetLastError(); cudaCheckError(error, __FILE__, __LINE__);

        error = cudaMemcpy(&computed_min, d_min, sizeof(int), cudaMemcpyDeviceToHost); cudaCheckError(error, __FILE__, __LINE__);
    }
    
    error = cudaFree(d_arr); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaFree(d_min); cudaCheckError(error, __FILE__, __LINE__);
    
    *m = computed_min;
}


__global__ void min_parallel(int* arr, const size_t size, int* block_mins)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t local_tid = threadIdx.x;

    size_t zero_idx = blockIdx.x * blockDim.x;

    extern __shared__ int shared_arr[];

    while (zero_idx < size)
    {
        shared_arr[local_tid] = tid < size ? arr[tid] : INT_MAX;
        __syncthreads();

        for (size_t i = blockDim.x / 2; i > 0; i = i / 2)
        {
            if (local_tid < i)
            {
                shared_arr[local_tid] = shared_arr[local_tid] < shared_arr[local_tid + i] ? shared_arr[local_tid] : shared_arr[local_tid + i];
            }
            __syncthreads();
        }

        zero_idx += stride;
        tid += stride;

        __syncthreads();
        if (local_tid == 0)
        {
            block_mins[blockIdx.x] = block_mins[blockIdx.x] < shared_arr[0] ? block_mins[blockIdx.x] : shared_arr[0];
        }
        __syncthreads();
    }
}


__host__ void run_min_pessimistic(int* arr, const size_t size, int* m)
{
    const size_t blocks = NUM_BLOCKS;
    const size_t threads = NUM_THREADS;
    
    int* d_arr;
    int* d_block_mins;

    int max = INT_MAX;

    
    int block_mins[blocks];
    #pragma omp parallel for
    for (size_t i = 0; i < blocks; ++i)
    {
        block_mins[i] = max;
    }

    cudaError_t error;

    // Allocate memory on the device
    error = cudaMalloc(&d_arr, size * sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMalloc(&d_block_mins, blocks * sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);

    // Copy the array to the device
    error = cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMemcpy(d_block_mins, block_mins, blocks * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);

    // Call the kernel
    min_parallel<<<blocks, threads, threads * sizeof(int)>>> (d_arr, size, d_block_mins);
    error = cudaDeviceSynchronize(); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaGetLastError(); cudaCheckError(error, __FILE__, __LINE__);
    
    // Copy the result back to the host
    error = cudaMemcpy(block_mins, d_block_mins, blocks * sizeof(int), cudaMemcpyDeviceToHost); cudaCheckError(error, __FILE__, __LINE__);

    // And extract the minimum of the array
    int minimum = block_mins[0];
    for (size_t i = 1; i < blocks; ++i)
    {
        if (block_mins[i] < minimum)
        {
            minimum = block_mins[i];
        }
    }

    error = cudaFree(d_arr); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaFree(d_block_mins); cudaCheckError(error, __FILE__, __LINE__);

    *m = minimum;
    return;
}

__global__ void min_atomic(int* arr, const size_t size, int* m)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < size; i = i + stride)
    {
        atomicMin(m, arr[i]);
    }
    return;
}

__host__ void run_min_atomic(int* arr, const size_t size, int* m)
{
    int* d_arr;
    int* d_min;
    int max = INT_MAX;

    cudaError_t error;
    error = cudaMalloc(&d_arr, size * sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMalloc(&d_min, sizeof(int)); cudaCheckError(error, __FILE__, __LINE__);

    error = cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaMemcpy(d_min, &max, sizeof(int), cudaMemcpyHostToDevice); cudaCheckError(error, __FILE__, __LINE__);

    min_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_arr, size, d_min);
    error = cudaDeviceSynchronize(); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaGetLastError(); cudaCheckError(error, __FILE__, __LINE__);

    error = cudaMemcpy(m, d_min, sizeof(int), cudaMemcpyDeviceToHost); cudaCheckError(error, __FILE__, __LINE__);

    error = cudaFree(d_arr); cudaCheckError(error, __FILE__, __LINE__);
    error = cudaFree(d_min); cudaCheckError(error, __FILE__, __LINE__);

}

int main(int argc, char** argv) 
{

    cudaDeviceGetAttribute(&NUM_BLOCKS, cudaDevAttrMultiProcessorCount, 0);
    std::cout << "NUM_BLOCKS: " << NUM_BLOCKS << std::endl;

    int* arr;
    size_t array_size;
    int computed_min;
    int computed_min_pessimistic;

    int generated_min;

    srand(time(NULL));

    array_size = atoll(argv[1]);
    arr = (int*) std::malloc(array_size * sizeof(int));

    assert(arr && "COULD NOT ALLOCATE MEMORY ON HOST");

    arr[0] = rand();
    generated_min = arr[0];

    #pragma omp parallel for
    for (size_t i = 1; i < array_size; ++i)
    {
        arr[i] = rand() % INT_MAX;
    }
    
    #pragma omp parallel for reduction(min:generated_min)
    for (size_t i = 1; i < array_size; ++i)
    {
        if (arr[i] < generated_min)
        {
            generated_min = arr[i];
        }
    }

    // Initialize the CUDA context
    cudaFree(0);

    std::cout << "Generated min " << generated_min << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_min_fixpoint(arr, array_size, &computed_min);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Computed min (fixpoint) " << computed_min << std::endl;
    std::cout << "Time: " << elapsed.count() << " milliseconds" << std::endl;
    
    auto start_pessimistic = std::chrono::high_resolution_clock::now();
    run_min_pessimistic(arr, array_size, &computed_min_pessimistic);
    auto end_pessimistic = std::chrono::high_resolution_clock::now();
    
    auto elapsed_pessimistic = std::chrono::duration_cast<std::chrono::milliseconds>(end_pessimistic-start_pessimistic);
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Computed min (pessimistic) " << computed_min_pessimistic << std::endl;
    std::cout << "Time: " << elapsed_pessimistic.count() << " milliseconds" << std::endl;

    auto start_atomic = std::chrono::high_resolution_clock::now();
    run_min_atomic(arr, array_size, &computed_min);
    auto end_atomic = std::chrono::high_resolution_clock::now();
    auto elapsed_atomic = std::chrono::duration_cast<std::chrono::milliseconds>(end_atomic-start_atomic);
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Computed min (atomic) " << computed_min << std::endl;
    std::cout << "Time: " << elapsed_atomic.count() << " milliseconds" << std::endl;


    std::free(arr);    
    return 0;



}
