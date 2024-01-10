#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"

// Malloc equivalents
__host__ void* cudaMallocSafe(size_t size);
__host__ void* cudaMallocManagedSafe(size_t size, unsigned int flags = cudaMemAttachGlobal);

// Free equivalents
__host__ void cudaFreeSafe(void* devPtr);