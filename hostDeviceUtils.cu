#include <iostream>
#include <string>

/**
 * Converts a size_t array (in host memory) to a size_t array (in device memory)
 * Also calls delete[] on the host memory version for you. You have to call cudaFree() on your own, though.
 */
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>>
void convertArrToCuda(T* &arr, size_t arrSize){
    T* deviceArr;
    cudaMalloc(&deviceArr, arrSize * sizeof(T));
    cudaMemcpy(deviceArr, arr, arrSize * sizeof(T), cudaMemcpyHostToDevice);
    delete[] arr;
    arr = deviceArr;
}

/**
 * Converts bundle data into a 1D array created using cudaMalloc
 * @param bundleData Bundle data. Each element in the array is a bundle, represented by another array terminated by -1.
 *  Note that each bundle need not be the same size. (Hence the -1 termination.)
 * @param numBundles The number of bundles.
 * @param[out] deviceBundles Output: The bundleData, represented as a 1D array. Basically, a flattened version of bundleData.
 * @param[out] bundleIndices The index (in bundleSeries) of each bundle. Note that bundleIndices[0] = 0 since the first
 *  bundle will obviously start at the first index of bundleSeries.
 */
void copyBundlesToDevice(size_t** bundleData, size_t numBundles, size_t* &deviceBundles, size_t* &bundleIndices){
    // First, we need to convert bundleData into a 1D array.
    // In other words, we need to flatten it.
    // First, find out how big the flattened array will be.
    // While we're doing this, we can also find out the indices of each bundle.
    size_t flattenedSize = 0;
    bundleIndices = new size_t[numBundles];
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        // The start of this bundle is at index flattenedSize
        bundleIndices[bundleNum] = flattenedSize;

        size_t bundleIdx = 0;
        while(bundleData[bundleNum][bundleIdx] != -1){
            // not the end
            // so we need to add this to the flattened list
            flattenedSize++;
            // and check if the next one is the end
            bundleIdx++;
        }
        // and one more for the -1 position.
        flattenedSize++;
        // bundleIdx++;
    }

    // Now we can construct our 1D array
    deviceBundles = new size_t[flattenedSize];
    size_t deviceBundleIdx = 0;
    // Copy each array over to their respective position...
    // Normally I'd just figure out memcpy but, well, no trust.
    // Also I haven't formally been taught memcpy so I don't trust my knowledge of it.
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        if(bundleIndices[bundleNum] != deviceBundleIdx){
            // this bundle isn't starting in the right spot.
            throw std::logic_error("A bundle isn't starting in the right spot.");
        }
        size_t bundleIdx = 0;
        while(bundleData[bundleNum][bundleIdx] != -1){
            deviceBundles[deviceBundleIdx] = bundleData[bundleNum][bundleIdx];
            deviceBundleIdx++;
            bundleIdx++;
        }
        // and then copy over the -1
        deviceBundles[deviceBundleIdx] = -1;
        deviceBundleIdx++;
    }
    if(deviceBundleIdx != flattenedSize){
        throw std::logic_error("The flattened list isn't the right size.");
    }

    // Finally, CUDAify bundleSeries and bundleIndices
    convertArrToCuda(bundleIndices, numBundles);
    convertArrToCuda(deviceBundles, flattenedSize);
}

/**
 * Converts series data into a 1D array created using cudaMalloc.
 *  This is different to copyBundlesToDevice because the trailing -1s at the end of each bundle are NOT copied
 *  Also, since we know this is a rectangular array, many optimizations can be done.
 * @param seriesData Series data. Note that this is a rectangular array of numSeries height and 2 items wide.
 * @param numSeries Number of series.
 * @param[out] deviceSeries Series data. This is made using cudaMalloc, and is a 1D array.
 */
void copySeriesToDevice(size_t** seriesData, size_t numSeries, size_t* &deviceSeries){
    // First, create the 1D array on host.
    deviceSeries = new size_t[numSeries*2];
    for(size_t s = 0; s < numSeries; s++){
        deviceSeries[(2*s)] = seriesData[s][0];
        deviceSeries[(2*s)+1] = seriesData[s][1];
    }
    // Convert to CUDA.
    convertArrToCuda(deviceSeries, 2*numSeries);
    // huh, that was easier than the bundles.
}