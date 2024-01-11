#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include "constants.cuh"

// Converts a vector to an array.
// Warning: You WILL need to delete[] this array at some point in the future to avoid memory leaks.
__host__ groupType* vectorToArray(thrust::host_vector<groupType> vec);