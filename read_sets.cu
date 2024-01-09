#include <fstream>
#include <thrust/host_vector.h>
#include "constants.cu"

// Note to self: Use thrust
// https://docs.nvidia.com/cuda/thrust/index.html

void readFile() {
	// Probably not the best method, but it's the best way to avoid exposing the structure of my computer
	// and also makes it so this wont break if i re-build this on a different computer... assuming i have working-data.
	std::string fileName = std::string(__FILE__) + "\\..\\working-data\\series.txt";

	std::ifstream file;
	file.open(fileName);

	std::string line;

	// We can't have a non-rectangular 2D array in device memory
	// but we can have a very long 1D array and then make a note of where each row starts and stops.
	// so if we have [[1,2], [3,4,5,6,7], [8], [9,10]], we would make note of the 1, 3, 8, and 9.
	thrust::host_vector<setType> data;
	thrust::host_vector<setType> row_indices;

	while (std::getline(file, line)) {
		std::cout << line << "\n";
	}

}