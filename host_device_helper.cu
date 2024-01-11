#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error_handler.cuh"
#include "constants.cuh"
#include <iostream>

// It's possible that these could be inlined safely
// However until it becomes an issue I don't think I'm gonna do that.
// The compiler will probably do it for me, but at least then it's the compiler being a million times smarter than I am.

__host__ void* cudaMallocSafe(size_t size) {
	void* dev_out;
	cudaError_t err = cudaMalloc(&dev_out, size);
	CUDAErrorCheck(err);
	// printf("Allocating memory of size %u\n", size);
	return dev_out;
}

__host__ void* cudaMallocManagedSafe(size_t size, unsigned int flags = cudaMemAttachGlobal) {
	void* dev_out;
	cudaError_t err = cudaMallocManaged(&dev_out, size, flags);
	CUDAErrorCheck(err);
	return dev_out;
}

__host__ void cudaFreeSafe(void* devPtr) {
	cudaError_t err = cudaFree(devPtr);
	CUDAErrorCheck(err);
}

__host__ groupNum* hostArrayToDevice(groupNum* arr, int size)
{
	groupNum* ret = (groupNum*)cudaMallocSafe(size * sizeof(groupNum));
	cudaMemcpy(ret, arr, size * sizeof(groupNum), cudaMemcpyHostToDevice);
	return ret;
}