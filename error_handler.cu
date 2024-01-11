#include "cuda_runtime.h"
#include <iostream>

__host__ void CUDAErrorCheck(cudaError_t error) {
	if (error == cudaSuccess) {
		// No error.
		return;
	}

	// Error detected!
	// Can __FILE__ access the parent file?
	// probably not tbh, it's a compile-time thing
	// Well, if it becomes a problem I investigate __FILE__ harder
	std::cerr << "CUDA Error detected: " << cudaGetErrorString(error);
	exit(error);
}