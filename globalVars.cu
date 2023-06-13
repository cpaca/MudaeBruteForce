#ifndef MUDAEBRUTEFORCE_GLOBALVARS
#define MUDAEBRUTEFORCE_GLOBALVARS
#include "types.cu"
// Score to beat.
__device__ size_t bestScore = 0;

// For each bundle, what series are in it?
// (Index 0 is also the bundle's size.)
__device__ size_t* bundleSeries = nullptr;
// Index of each bundle in bundleSeries. So bundleSeries[bundleIndices[n]] is the first index of a bundle in bundleSeries.
__device__ size_t* bundleIndices = nullptr;

// For each set, what bundles contain it?
// The format is... kind of a long description.
// First, let setBundlesSetSize = (numBundles/sizeof(size_t))
// And for shorthand, let sBSS = setBundlesSetSize
// Indices setBundles[setNum * sBSS] to setBundles[(setNum+1)*sBSS - 1] are the indices for set setNum
// In other words, to loop over all values in setBundles relevant to a set:
// for(int i = 0; i < sBSS; i++){/*do something with setBundles[setNum*sBSS + i]*/}
//
// Now express setBundles[0], setBundles[1], ... as a bitstream.
// The first bit represents if the set is in bundle # 0
// The second bit represents if the set is in bundle # 1
// etc.
// Because this is a bitstream and size_t is 64-bits:
// the 65th bit (aka, the first bit of setBundles[1], aka setBundles[1]&0) represents if the set is in bundle #65
//
// Note that this is setBundles, so it needs to work for all SETS. Even Bundles.
// Also note that for bundles, their "bitstream" is all 0s except for itself, where it is 1.
__device__ size_t* setBundles = nullptr;
__constant__ size_t setBundlesSetSize = -1; // note that setBundles[-1] = illegal (unsigned type)

// Data about each series.
// deviceSeries[2n] is the size of series n
// deviceSeries[2n+1] is the value of series n
__device__ size_t* deviceSeries = nullptr;

// Free bundles.
// If freeBundles[n] is non-zero, then bundle n is free.
__device__ size_t* freeBundles = nullptr;

// The size of each set.
// Note that this is setSize_t, not size_t.
// This is important because of byte limitations.
__device__ setSize_t* global_setSizes = nullptr;
extern __shared__ setSize_t setSizes[];

// The order to delete each set in.
__device__ size_t* setDeleteOrder = nullptr;

// Turns out if I do this and #include this file, it works fine.
// Maximum number of bundles/series that can be activated.
const std::uint32_t MAX_DL = 50;
// Maximum number of free bundles.
// Can be changed whenever, but keep it low or CUDA will demand much more memory than necessary.
const std::uint32_t MAX_FREE_BUNDLES = 5;
// Overlap limit, defined in Mudae
const std::uint32_t OVERLAP_LIMIT = 30000;
// How many blocks to run.
// Note that each block gets 512 threads.
#define NUM_BLOCKS (1 << 12)
// "MinSize" is a variable determining the minimum size a series needs to be to be added to the DL.
// MinSize gets divided by 2 while the remainingOverlap exceeds minSize, so even a minSize of 2^31 will get fixed
// down to remainingOverlap levels.
// MAX_MINSIZE determines the maximum value minSize can be.
const std::uint32_t MAX_MINSIZE = 100;

#endif