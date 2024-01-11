#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"
#include <string>

__host__ std::string getToken(std::string& str);
__host__ void readFile();