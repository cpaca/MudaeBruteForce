#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include "constants.cuh"

__host__ groupType* vectorToArray(thrust::host_vector<groupType> vec)
{
	auto size = vec.size();
	groupType* out = new groupType[size];

	for (int i = 0; i < vec.size(); i++) {
		out[i] = vec[i];
	}
	
	return out;
}