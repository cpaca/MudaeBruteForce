#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.cuh"
#include "host_device_helper.cuh"
#include <iostream>

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
	/*
	for (int i = 0; i < dev_numRows; i++) {
		groupNum rowStart = dev_rowIndices[i]; // inclusive
		groupNum rowEnd = dev_rowIndices[i + 1]; // exclusive

		printf("Row data: ");
		for (int j = rowStart; j < rowEnd; j++) {
			printf("%u ", dev_groupData[j]);
		}
		printf("\n");

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

__host__ groupNum* hostArrayToDevice(groupNum* arr, int size)
{
	groupNum* ret = (groupNum*)cudaMallocSafe(size * sizeof(groupNum));
	cudaMemcpy(ret, arr, size * sizeof(groupNum), cudaMemcpyHostToDevice);
	return ret;
}

__host__ void saveGroupData(groupNum* groupData, groupNum* rowIndices, groupNum numRows)
{
	// Array validation host-side:

	/*
	std::cout << "Reconstructing groupData..." << "\n";

	for (int i = 0; i < numRows; i++) {
		groupNum startIdx = rowIndices[i];
		groupNum endIdx = rowIndices[i + 1];
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
	auto devClone_rowIndices = hostArrayToDevice(host_rowIndices, host_numRows);

	cudaMemcpyToSymbol(dev_groupData, &devClone_groupData, sizeof(devClone_groupData));
	cudaMemcpyToSymbol(dev_rowIndices, &devClone_rowIndices, sizeof(devClone_rowIndices));
	cudaMemcpyToSymbol(dev_numRows, &host_numRows, sizeof(host_numRows));

	groupDataDeviceValidate<<<1, 1 >>>();
}