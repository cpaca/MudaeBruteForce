#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error_handler.cu"

// It's possible that these could be inlined safely
// However until it becomes an issue I don't think I'm gonna do that.
// The compiler will probably do it for me, but at least then it's the compiler being a million times smarter than I am.

/// <summary>
/// Similar to cudaMalloc, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// </summary>
/// <param name="size">Number of bytes to allocate.</param>
/// <returns></returns>
__host__ void* cudaMallocSafe(size_t size) {
	void* dev_out;
	cudaError_t err = cudaMalloc(&dev_out, size);
	CUDAErrorCheck(err);
	return dev_out;
}

/// <summary>
/// Similar to cudaMallocManaged, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// <para/> The "flags" parameter is still available if I ever need it, but looking at documentation, I highly doubt I will.
/// </summary>
/// <param name="size">Number of bytes to allocate.</param>
/// <returns></returns>
__host__ void* cudaMallocManagedSafe(size_t size, unsigned int flags = cudaMemAttachGlobal) {
	void* dev_out;
	cudaError_t err = cudaMallocManaged(&dev_out, size, flags);
	CUDAErrorCheck(err);
	return dev_out;
}

/// <summary>
/// Similar to cudaFree, however the error-handling is automatically done.
/// This means that the signature can be treated more like ordinary malloc.
/// </summary>
/// <param name="size"></param>
/// <returns></returns>
__host__ void cudaFreeSafe(void* devPtr) {
	cudaError_t err = cudaFree(devPtr);
	CUDAErrorCheck(err);
}