#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"
#include "constants.cuh"

// Malloc equivalents

/// <summary>
/// Similar to cudaMalloc, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// </summary>
/// <param name="size">Number of bytes to allocate.</param>
/// <returns></returns>
__host__ void* cudaMallocSafe(size_t size);

/// <summary>
/// Similar to cudaMallocManaged, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// <para/> The "flags" parameter is still available if I ever need it, but looking at documentation, I highly doubt I will.
/// </summary>
/// <param name="size">Number of bytes to allocate.</param>
/// <returns></returns>
__host__ void* cudaMallocManagedSafe(size_t size, unsigned int flags = cudaMemAttachGlobal);

// Free equivalents

/// <summary>
/// Similar to cudaFree, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// </summary>
/// <param name="size"></param>
/// <returns></returns>
__host__ void cudaFreeSafe(void* devPtr);

// Misc functions

// Yes, I have code in a header function.
// No, I don't care. See this StackOverflow question which lists 3 solutions. This is the first solution.
// https://stackoverflow.com/a/456716
// The other two solutions are crossed out and won't work for me, either.
/// <summary>
/// Copies an array from host-side to device-side memory.
/// </summary>
/// <param name="arr">Pointer to the start of the array</param>
/// <param name="size">Size of the array</param>
/// <returns></returns>
template<typename T>
__host__ T* hostArrayToDevice(T* arr, int size) {
	T* ret = (T*)cudaMallocSafe(size * sizeof(T));
	cudaMemcpy(ret, arr, size * sizeof(T), cudaMemcpyHostToDevice);
	return ret;
}
