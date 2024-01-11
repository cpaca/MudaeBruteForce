#pragma once

// This file should exclusively contain global variables.
// I know about the practices regarding global variables... I also know that I genuinely will need things like groupData EVERYWHERE in my program.
// Like, almost without exception. At the very least, I put comments to reduce the impact that having global variables has.

#include "cuda_runtime.h"
#include "constants.cuh"

// Note: These variables should all be write-once, and that one write should be in read_sets.cu readLines(), which should also be called only once.
extern groupType* host_groupData;
extern groupType* host_rowIndices;
extern groupType host_numRows;