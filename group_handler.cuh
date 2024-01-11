#pragma once

#include "cuda_runtime.h"
#include "constants.cuh"

// Sets and saves the data for groupHandler.
// Should only be called once.
__host__ void saveGroupData(groupType* groupData, groupType* rowIndices, groupType numRows);