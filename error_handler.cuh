#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"

__host__ void CUDAErrorCheck(cudaError_t code);