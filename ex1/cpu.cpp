/**
 * Sequential implementation of the algorithm to find a global minimum in an array. I use the positive integers as randomly generated numbers.
 * The minimum has instead the value -1. The array is constructed in a way that the minimum is unique.
 * 
 * usage:
 * ./sequential ARRAY_SIZE
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <chrono>


int min(int* arr, const int size, int* arr_min)
{
    int idx = 0;
    (*arr_min) = arr[idx]; 
    for (int i = 1; i < size; ++i)
    {
        if (arr[i] < *arr_min)
        {
            (*arr_min) = arr[i];
            idx = i;
        }
    }

    return idx;
}

int main(int argc, char** argv) 
{
    int* arr;
    int array_size;
    int computed_min_idx;
    int computed_min;

    int generated_min;
    int generated_min_idx;
    


    srand(time(NULL));

    array_size = atoi(argv[1]);
    arr = (int*) std::malloc(array_size * sizeof(int));

    assert(arr && "COULD NOT ALLOCATE MEMORY");

    arr[0] = rand();
    generated_min = arr[0];
    generated_min_idx = 0;

    for (int i = 0; i < array_size; ++i)
    {
        arr[i] = rand();
        if(arr[i] < generated_min)
        {
            generated_min = arr[i];
            generated_min_idx = i;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    computed_min_idx = min(arr, array_size, &computed_min);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

    assert(computed_min_idx == generated_min_idx && computed_min == generated_min && "!! IMPLEMENTATION ERROR !!");

    std::cout << "Generated min " << generated_min << " at index " << generated_min_idx << std::endl;
    std::cout << "Computed min " << computed_min << " at index " << computed_min_idx << std::endl;
    std::cout << "Time: " << elapsed.count() << " microseconds" << std::endl;

    std::free(arr);    
    return 0;

}

