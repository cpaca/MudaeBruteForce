#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.cuh"
#include <iostream>

groupType* host_groupData = nullptr;
groupType* host_rowIndices = nullptr;
groupType host_numRows = 0;
__device__ groupType* dev_groupData = nullptr;
__device__ groupType* dev_rowIndices = nullptr;
__device__ groupType dev_numRows = 0;

__host__ void saveGroupData(groupType* groupData, groupType* rowIndices, groupType numRows)
{
	// Array validation host-side:

	/*
	std::cout << "Reconstructing groupData..." << "\n";

	for (int i = 0; i < numRows; i++) {
		groupType startIdx = rowIndices[i];
		groupType endIdx = rowIndices[i + 1];
		for (int i = startIdx; i < endIdx; i++) {
			std::cout << groupData[i] << " ";
		}
		std::cout << "\n";
	}
	//*/
}