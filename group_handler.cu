#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.cuh"

groupType* host_groupData = nullptr;
groupType* host_rowIndices = nullptr;
groupType host_numRows = 0;
__device__ groupType* dev_groupData = nullptr;
__device__ groupType* dev_rowIndices = nullptr;
__device__ groupType dev_numRows = 0;

__host__ void saveGroupData(groupType* groupData, groupType* rowIndices, groupType numRows)
{

}