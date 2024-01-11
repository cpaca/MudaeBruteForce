#include "constants.cuh"

// See the CUH file for why I need global vars.
// But extern values need at least one static/global definition somewhere, and here it is. It get overridden and assert-checked later, though.

groupType* host_groupData = nullptr;
groupType* host_rowIndices = nullptr;
groupType host_numRows = 0;