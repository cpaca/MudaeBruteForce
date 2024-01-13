#pragma once
// This can also be thought of as the #define file.
#include <cstdint>

#ifdef __CUDA_ARCH__
#define globalFunc(funcName, ...) funcName<<<__VA_ARGS__>>>
#else
#define deadFunc(...) //
#define globalFunc(...) deadFunc
#endif

typedef std::uint32_t groupNum;