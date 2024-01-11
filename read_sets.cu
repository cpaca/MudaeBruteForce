#include <fstream>
#include <thrust/host_vector.h>
#include "constants.cuh"
#include "vector_helper.cuh"

// Note to self: Use thrust
// https://docs.nvidia.com/cuda/thrust/index.html

/// <summary>
/// Similar to strtok(str, "$"), but strtok only accepts char* instead of std::string.
/// Realistically I could just write this in a way that uses strtok... but whatever.
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
__host__ std::string getToken(std::string& str) {
	auto idx = str.find("$");

	std::string token;
	if (idx == std::string::npos) {
		// didn't find delimiter
		// therefore no more delimiters
		token = str;
		str = "";
	}
	else {
		token = str.substr(0, idx);
		str = str.substr(idx + 1);
	}
	return token;
}

__host__ void readFile() {
	// Probably not the best method, but it's the best way to avoid exposing the structure of my computer
	// and also makes it so this wont break if i re-build this on a different computer... assuming i have working-data.
	std::string fileName = std::string(__FILE__) + "\\..\\working-data\\series.txt";

	std::ifstream file;
	file.open(fileName);

	std::string line;

	// We can't have a non-rectangular 2D array in device memory
	// but we can have a very long 1D array and then make a note of where each row starts and stops.
	// so if we have [[1,2], [3,4,5,6,7], [8], [9,10]], we would make note of the 1, 3, 8, and 9.
	thrust::host_vector<groupType> groupData;
	thrust::host_vector<groupType> rowIndices;

	while (std::getline(file, line)) {
		// Validation that readline works
		// std::cout << line << "\n";
		
		// Start of a new row.
		auto size = groupData.size();
		rowIndices.push_back(size);

		while (!line.empty()) {
			auto token = getToken(line);

			groupType num = (groupType) stoi(token);
			groupData.push_back(num);
		}
	}
	// Add one more so the very-last row knows where it ends.
	auto size = groupData.size();
	// If any of the elements of rowIndices exceeds 2billion, then this one will also exceed 2billion.
	// The only exception would be if there's over 9 quintillion elements in groupData, but for that to happen without an out-of-memory exception
	// would require approximately 36 exabytes of RAM.
	if (size > 2000000000) {
		// over 2 billion... might overload groupType.
		// I'm pretty sure this will never happen, but better to have the check.
		std::cerr << "SetType is going to get overloaded. Make it a bigger type.";
		exit(1);
	}
	rowIndices.push_back(size);

	// These vectors will never, ever be modified again (may be read again)
	// So shrink them to min size...
	groupData.shrink_to_fit();
	rowIndices.shrink_to_fit();

	groupType* host_groupData = vectorToArray(groupData);
	groupType* host_rowIndices = vectorToArray(rowIndices);
	groupType numRows = rowIndices.size();

	// Validation that vectorToArray works on host-side:

	std::cout << "Reconstructing groupData..." << "\n";

	for (int i = 0; i < numRows - 1; i++) {
		groupType startIdx = host_rowIndices[i];
		groupType endIdx = host_rowIndices[i + 1];
		for (int i = startIdx; i < endIdx; i++) {
			std::cout << host_groupData[i] << " ";
		}
		std::cout << "\n";
	}
}