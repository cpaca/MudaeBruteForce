#pragma once

#include "cuda_runtime.h"
#include "constants.cuh"

// Sets and saves the data for groupHandler.
// Should only be called once.
__host__ void saveGroupData(groupType* groupData, groupType* rowIndices, groupType numRows);

/// <summary>
/// Validates the device-side global-variables written in readFile. These are groupData, rowIndices, and numRows.
/// This is a device-side function, but is meant to be run with 1-thread since the validation is done by looking at the numbers visually.
/// </summary>
__global__ void groupDataDeviceValidate();