#pragma once

// Included so that __host__ gets the correct code-sense coloring.
#include "cuda_runtime.h"
#include <string>

/// <summary>
/// Similar to strtok(str, "$"), but strtok only accepts char* instead of std::string.
/// Realistically I could just write this in a way that uses strtok... but whatever.
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
__host__ std::string getToken(std::string& str);
__host__ void readFile();