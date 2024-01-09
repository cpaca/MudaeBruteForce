// Credit to this StackOverflow question and answer for pretty much this entire file.
// I change out some bits into formats that I like more or to make it work properly (like the #includes)
// but I change it so little that I cannot take credit for this work.
// https://stackoverflow.com/a/14038590
// Accessed January 8, 2023
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define CUDAErrorCheck(ans) { CUDAAssert((ans), __FILE__, __LINE__); }

inline void CUDAAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        // Changed this line out because it wasn't playing nicely
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
        if (abort) exit(code);
    }
}