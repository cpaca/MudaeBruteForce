#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.cuh"
#include "host_device_helper.cuh"
#include "group_handler.cuh"
#include <iostream>
#include <cassert>

groupNum* host_groupData = nullptr;
groupNum* host_rowIndices = nullptr;
groupNum host_numRows = 0;
__device__ groupNum* dev_groupData = nullptr;
__device__ groupNum* dev_rowIndices = nullptr;
__device__ groupNum dev_numRows = 0;

// I understand that these functions should logically be at the bottom of the file, not near the top,
// but when I put it near the bottom, I can't compile.
// It's weird, but this is easier than finding a fix.
__global__ void groupDataDeviceValidate() {
	// I understand I could just do this in one for-loop
	// but this is a more "genuine" representation of what each row represents
	//*
	for (int i = 0; i < dev_numRows; i++) {
		groupNum rowStart = dev_rowIndices[i]; // inclusive
		groupNum rowEnd = dev_rowIndices[i + 1]; // exclusive

		printf("Row #%4u data:", i+1);
		for (int j = rowStart; j < rowEnd; j++) {
			printf(" %u", dev_groupData[j]);
		}
		printf(", from idx %u to idx %u\n", rowStart, rowEnd);

		// ... visual studio is too dumb so only have nvcc process this part
#ifdef __CUDA_ARCH__
		// Without this the printf buffer gets too filled up.
		// Performance isn't really necessary here since we just do a visual check.
		__nanosleep(1000000);
#endif
	}
	printf("End of groupDataDeviceValidate\n");
	//*/
}

__host__ __device__ void getGroupDataValidate() {
	groupNum numRows;
#ifdef __CUDA_ARCH__
	numRows = dev_numRows;
#else
	numRows = host_numRows;
#endif
	for (int i = 1; i <= numRows; i++) {
		groupType data = getGroupData(i);
		printf("Group %u has an efficiency of %u/%u and %u bundles\n", i, data.value, data.weight, data.numBundles);
#ifdef __CUDA_ARCH__
		__nanosleep(1000000);
#endif
	}
}

__global__ void getGroupDataDeviceValidate() {
	getGroupDataValidate();
}

__host__ __device__ groupType getGroupData(groupNum numGroup)
{
	// Like the comment said, 1-indexed.
	groupNum rowNum = numGroup - 1;

	// Technically this variable renaming procedure isn't necessary
	// but it reduces a LOT of code reuse and also helps visual studio at least somewhat understand what's going on.
	groupNum* groupData;
	groupNum* rowIndices;
	groupNum numRows;

#ifdef __CUDA_ARCH__
	// Device side
	groupData = dev_groupData;
	rowIndices = dev_rowIndices;
	numRows = dev_numRows;
#else
	// Host side
	groupData = host_groupData;
	rowIndices = host_rowIndices;
	numRows = host_numRows;
#endif

	// exists in device cuda natively
	// and in host code with cassert
	assert(rowNum < numRows);
	assert(rowNum >= 0);

	groupNum dataIdx = rowIndices[rowNum];
	groupNum nextIdx = rowIndices[rowNum + 1];
	groupNum* dataPtr = groupData + dataIdx;

	groupType out;

	out.weight = dataPtr[0];
	out.value = dataPtr[1];
	out.numBundles = nextIdx - dataIdx - 2;
	out.bundles = dataPtr + 2;

	return out;
}

__host__ void saveAllGroupData(groupNum* groupData, groupNum* rowIndices, groupNum numRows)
{
	// Array validation host-side:

	/*
	std::cout << "Reconstructing groupData..." << "\n";

	for (int i = 0; i < numRows; i++) {
		groupNum startIdx = rowIndices[i];
		groupNum endIdx = rowIndices[i + 1];
		std::cout << "Row data: ";
		for (int i = startIdx; i < endIdx; i++) {
			std::cout << groupData[i] << " ";
		}
		std::cout << "\n";
	}
	//*/

	host_groupData = groupData;
	host_rowIndices = rowIndices;
	host_numRows = numRows;

	// I understand that this is "sort of" a 2D array so theoretically, I should use cudaMallocPitch, however:
	// - This is NOT a proper 2D array; the first row always has 2 elements and the last (many) rows always have at least 3 elements
	// --- (Also, some rows may have 4, 5, 6, etc. elements, so this is even less of a 2D array)
	// --- so cudaMallocPitch probably wouldn't function properly here
	// Basically, as I understand it, cudaMallocPitch is built for rectangular 2D arrays
	// This isn't rectangular.
	auto devClone_groupData = hostArrayToDevice(host_groupData, host_rowIndices[host_numRows]);
	// +1 because the element after the last row exists so the last row knows when to stop
	auto devClone_rowIndices = hostArrayToDevice(host_rowIndices, host_numRows + 1); 

	cudaMemcpyToSymbol(dev_groupData, &devClone_groupData, sizeof(devClone_groupData));
	cudaMemcpyToSymbol(dev_rowIndices, &devClone_rowIndices, sizeof(devClone_rowIndices));
	cudaMemcpyToSymbol(dev_numRows, &host_numRows, sizeof(host_numRows));

	// groupDataDeviceValidate<<<1, 1 >>>();

	// getGroupDataValidate();
	// getGroupDataDeviceValidate<<<1, 1 >>>();
}

__host__ void cleanupGroupData()
{
	delete[] host_groupData;
	delete[] host_rowIndices;
	groupNum* toDelete;
	cudaMemcpyFromSymbol(&toDelete, dev_groupData, sizeof(toDelete));
	cudaFreeSafe(toDelete);
	cudaMemcpyFromSymbol(&toDelete, dev_rowIndices, sizeof(toDelete));
	cudaFreeSafe(toDelete);
}