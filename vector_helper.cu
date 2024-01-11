#include "cuda_runtime.h"
#include <thrust/host_vector.h>
#include "constants.cuh"

__host__ groupNum* vectorToArray(thrust::host_vector<groupNum> vec)
{
	auto size = vec.size();
	groupNum* out = new groupNum[size];

	for (int i = 0; i < vec.size(); i++) {
		out[i] = vec[i];
	}
	
	return out;
}