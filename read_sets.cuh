#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"
#include "constants.cuh"
#include <string>

/// <summary>
/// Similar to strtok(str, "$"), but strtok only accepts char* instead of std::string.
/// Realistically I could just write this in a way that uses strtok... but whatever.
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
__host__ std::string getToken(std::string& str);
__host__ void readFile();

/// <summary>
/// Clones an array located in host-memory and puts it in device (GPU) memory. This saves the memory in device memory (aka not managed memory)
/// Note this creates a clone that needs to be cudaFree'd (or cudaFreeSafe'd) later.
/// This function fits better in another file, but I don't know what else that file will have yet. For now, it goes here.
/// </summary>
/// <param name="arr"></param>
/// <returns></returns>
__host__ groupType* hostArrayToDevice(groupType* arr, int size);

/// <summary>
/// Validates the device-side global-variables written in readFile. These are groupData, rowIndices, and numRows.
/// This is a device-side function, but is meant to be run with 1-thread since the validation is done by looking at the numbers visually.
/// </summary>
__global__ void readFileDeviceValidate();