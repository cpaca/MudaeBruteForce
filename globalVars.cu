#ifndef MUDAEBRUTEFORCE_GLOBALVARS
#define MUDAEBRUTEFORCE_GLOBALVARS
#include "types.cu"
// For each bundle, what series are in it?
// (Index 0 is also the bundle's size.)
__constant__ size_t* bundleSeries = nullptr;
// Index of each bundle in bundleSeries. So bundleSeries[bundleIndices[n]] is the first index of a bundle in bundleSeries.
__constant__ size_t* bundleIndices = nullptr;

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
__constant__ size_t* setBundles = nullptr;
__constant__ size_t setBundlesSetSize = -1; // note that setBundles[-1] = illegal (unsigned type)
size_t host_setBundlesSetSize;

// Data about each series.
// deviceSeries[2n] is the size of series n
// deviceSeries[2n+1] is the value of series n
__constant__ size_t* deviceSeries = nullptr;

// Free bundles.
// If freeBundles[n] is non-zero, then bundle n is free.
__constant__ size_t* freeBundles = nullptr;

// The size of each set.
// Note that this is setSize_t, not size_t.
// This is important because of byte limitations.
__constant__ setSize_t* global_setSizes = nullptr;
extern __shared__ setSize_t setSizes[];

// The order to delete each set in.

size_t* host_setDeleteOrder;
size_t setDeleteIndex = 0;
__constant__ size_t expectedSetToDelete = 0;

// Queues
__device__ TaskQueue inTaskQueue;
__device__ TaskQueue outTaskQueue;
__device__ TaskQueue deadTaskQueue;
__constant__ size_t queuePitch;

#endif