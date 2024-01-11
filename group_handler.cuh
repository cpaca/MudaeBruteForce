#pragma once

#include "cuda_runtime.h"
#include "constants.cuh"

// Whenever group-data is needed, it's almost definitely certain all of this data will be needed.
struct groupType {
	groupNum weight;
	groupNum value;
	groupNum numBundles; // How many bundles are associated with this group. 0 if this group is a bundle (aka doesnt count itself)
	const groupNum* bundles; // Pointer to an element of groupData so DO NOT EDIT THIS!!!!!
};

// Sets and saves the data for groupHandler.
// Should only be called once.
__host__ void saveAllGroupData(groupNum* groupData, groupNum* rowIndices, groupNum numRows);

// Gets the data for one group (aka one row)
// WARNING: ONE-INDEXED
// SO GROUP 1 IS INDEX ZERO OF GROUPDATA
// This way, group 1 represents line 1 of the actual text file.
__host__ __device__ groupType getGroupData(groupNum numGroup);

/// <summary>
/// Validates variables: groupData, rowIndices, and numRows.
/// This is a device-side function, but is meant to be run with 1-thread since the **validation is done by looking at the numbers visually**!!!!.
/// </summary>
__global__ void groupDataDeviceValidate();

// Destructor for the group data, basically.
__host__ void cleanupGroupData();