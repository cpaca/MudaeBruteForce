#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
    size_t numBundles;
    // first read the bundlesStr
    std::string* bundlesStr = getLines("../working-data/bundle_data.txt", numBundles);
    auto** bundleData = new size_t*[numBundles];
    auto* bundleNames = new std::string[numBundles];
    for(size_t i = 0; i < numBundles; i++){
        bundleData[i] = getLineData(bundlesStr[i], bundleNames[i]);
    }
    delete[] bundlesStr;

    // doing some validation
    // comment out if you don't want that validation.
    /*
    for(size_t i = 0; i < numBundles; i++){
        std::cout << "Validating bundle " << std::to_string(i) << "\n";
        size_t* bundlePtr = bundleData[i];
        while(*bundlePtr != -1){
            std::cout << std::to_string(*bundlePtr) << ", ";
            bundlePtr++;
        }
        std::cout << "\n";
    }
    //*/

    // next read the seriesStr
    size_t numSeries;
    std::string* seriesStr = getLines("../working-data/series_data.txt", numSeries);
    auto** seriesData = new size_t*[numSeries];
    auto* seriesNames = new std::string[numSeries];
    for(size_t i = 0; i < numSeries; i++){
        seriesData[i] = getLineData(seriesStr[i], seriesNames[i]);
    }
    delete[] seriesStr;

    // validate that no bundlesStr are exceeding seriesID:
    // Validate bundles.
    for(size_t i = 0; i < numBundles; i++){
        auto expectedSize = bundleData[i][0];

        size_t idx = 1; // idx 0 is size so not a series
        size_t actualSize = 0;
        size_t lastSeriesId = 0;
        auto seriesID = bundleData[i][idx];
        while(seriesID != -1){
            if(seriesID >= numSeries){
                std::cerr << "Bundle has invalid SeriesID: " << bundleNames[i] << "\n";
                return 1;
            }
            if(seriesID < lastSeriesId){
                std::cerr << "Bundle series are not in sorted order: " << bundleNames[i] << "\n";
                return 3;
            }
            actualSize += seriesData[seriesID][0];
            idx++;
            lastSeriesId = seriesID;
            seriesID = bundleData[i][idx];
        }

        if(expectedSize != actualSize){
            std::cerr << "Bundle size is not equivalent to the sum of the sizes of its series.\n";
            std::cerr << "Violating BundleID: " << std::to_string(i);
            std::cerr << "Violating bundleName: " << bundleNames[i];
            return 2;
        }
    }

    // Initialize host_setBundlesSetSize since it's needed several times before setBundles actually gets initialized.
    // (Notably, the taskQueue needs it)
    // Explanation of *8: sizeof() returns size in bytes, I want size in bits.
    // Explanation of +1: If there are 7 bundles, setSize_t should be 1, not 0.
    // host_ added because I can't write device data directly @ host level.
    host_setBundlesSetSize = (numBundles / (sizeof(size_t) * 8)) + 1;

    // There are certain bundles which are free. Namely, Western and Real Life People, if you have togglewestern and toggleirl
    // There's also togglehentai on some servers.
    // So we should keep track of that.
    // This is size_t, even if it could be bool*, just for consistency.
    auto* host_freeBundles = new size_t[numBundles];
    std::string freeBundleNames[] = {"Western", "Real Life People"};
    int numFreeBundles = sizeof(freeBundleNames)/sizeof(freeBundleNames[0]);
    int freeBundlesFound = 0;
    for(size_t i = 0; i < numBundles; i++){
        std::string bundleName = bundleNames[i];
        host_freeBundles[i] = 0;
        for(const std::string& freeBundleName : freeBundleNames){
            if(freeBundleName == bundleName){
                host_freeBundles[i] = ~0; // not 0 = all 1s (0xfff...)
                freeBundlesFound += 1;
                break;
            }
        }
    }
    if(freeBundlesFound != numFreeBundles){
        std::cout << "Found " << std::to_string(freeBundlesFound) << " free bundles\n" << "Expected " <<
            std::to_string(numFreeBundles);
        throw std::logic_error("Missing some free bundles.");
    }
    // No new so no need for a delete on freeBundleNames.
    // And convert freeBundles into a CUDA usable form.
    initializeGlobalSetSizes(numSeries, numBundles, seriesData, bundleData, host_freeBundles);
    initSetDeleteOrder(host_freeBundles, bundleData, seriesData, numSeries, numBundles);
    initTaskQueue(host_freeBundles, bundleData, seriesData, numSeries, numBundles);
    convertArrToCuda(host_freeBundles, numBundles);
    if(host_freeBundles == nullptr){
        std::cout << "FreeBundles not initialized correctly.";
    }
    cudaMemcpyToSymbol(freeBundles, &host_freeBundles, sizeof(host_freeBundles));

    initializeSetBundles(numBundles, numSeries, bundleData, seriesData);

    // time to do CUDA.
    // https://forums.developer.nvidia.com/t/how-to-cudamalloc-two-dimensional-array/4042
    // Mother fucker, I'm gonna have to convert the bundleData into a 1D array.
    // At least I can use seriesData as a 2D array, using the same schema as before.
    // Can't do that for bundleData because bundleData is non-rectangular.
    // ... bleh, cudaMallocPitch is annoying, I might also do seriesData as a 1D array...

    size_t* host_bundleSeries = nullptr;
    size_t* host_bundleIndices = nullptr;
    copyBundlesToDevice(bundleData, numBundles, host_bundleSeries, host_bundleIndices);
    if(host_bundleSeries == nullptr || host_bundleIndices == nullptr){
        throw std::logic_error("Device bundles and/or bundleIndices did not get overwritten properly.");
    }
    cudaMemcpyToSymbol(bundleSeries, &host_bundleSeries, sizeof(host_bundleSeries));
    cudaMemcpyToSymbol(bundleIndices, &host_bundleIndices, sizeof(host_bundleIndices));

    size_t* host_deviceSeries = nullptr;
    copySeriesToDevice(seriesData, numSeries, host_deviceSeries);
    if(host_deviceSeries == nullptr){
        throw std::logic_error("Device series did not get overwritten properly.");
    }
    cudaMemcpyToSymbol(deviceSeries, &host_deviceSeries, sizeof(host_deviceSeries));

    knapsackInit();

    // Non-array values are available in Device memory. Proof: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    // Section 3.2.2 uses "int N" in both host and device memory.

    // https://stackoverflow.com/questions/23260074/allocating-malloc-a-double-in-cuda-device-function
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);

    size_t sharedMemoryNeeded = (numBundles + numSeries) * sizeof(setSize_t);

    // makeError<<<2, 512>>>(numBundles, numSeries);
    clock_t GPUTime = 0;
    clock_t CPUTime = 0;

    // std::cout << "Executing FindBest with " << std::to_string(NUM_BLOCKS) << " blocks of 512 threads each.\n";
    // findBest<<<NUM_BLOCKS, 512, sharedMemoryNeeded>>>(numBundles, numSeries);
    std::cout << "Shared memory needed: " << std::to_string(sharedMemoryNeeded) << "\n";
    // reminder to self: 40 blocks of 512 threads each
    // for some reason 1024 threads per block throws some sort of error
    cudaError_t syncError;
    for(size_t i = 0; i < 80; i++) {
        GPUTime -= clock();
        newFindBest<<<40, 512, sharedMemoryNeeded>>>(numBundles, numSeries);
        syncError = cudaDeviceSynchronize();
        GPUTime += clock();

        if(syncError != cudaSuccess){
            break;
        }

        CPUTime -= clock();
        reloadTaskQueue();
        CPUTime += clock();
    }

    printProfilingData();
    std::cout << "Time taken on GPU (seconds): " << std::to_string(GPUTime/(double)CLOCKS_PER_SEC) << "\n";
    std::cout << "Time taken on CPU (seconds): " << std::to_string(CPUTime/(double)CLOCKS_PER_SEC) << "\n";

    if (syncError != cudaSuccess) {
        std::cout << "cudaDeviceSync   invoked a CUDA error" << std::endl;
        const char *errName = cudaGetErrorName(syncError);
        printf("%s\n", errName);
        const char *errStr = cudaGetErrorString(syncError);
        printf("%s\n", errStr);
    }
    else{
        std::cout << "cudaDeviceSync   did not invoke a CUDA error" << std::endl;
    }

    cudaError_t lasterror = cudaGetLastError();
    if (lasterror != cudaSuccess) {
        std::cout << "cudaGetLastError found a CUDA error" << std::endl;
        const char *errName = cudaGetErrorName(lasterror);
        printf("%s\n", errName);
        const char *errStr = cudaGetErrorString(lasterror);
        printf("%s\n", errStr);
    }
    else{
        std::cout << "cudaGetLastError did not find a CUDA error" << std::endl;
    }

    printf("FindBest finished\n");

    // Free up memory.
    std::cout << "Freeing up memory.\n";
    // Free up the bundles.
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        delete[] bundleData[bundleNum];
    }
    delete[] bundleData;
    delete[] bundleNames;
    // Free up the series.
    for(size_t seriesNum = 0; seriesNum < numSeries; seriesNum++){
        delete[] seriesData[seriesNum];
    }
    delete[] seriesData;
    delete[] seriesNames;
    // Free up CUDA.
    // Well they're all __device__ variables now
    // https://forums.developer.nvidia.com/t/will-cuda-free-the-memory-when-my-application-exit/16113/5
    // and they'll get auto-freed when application exit.
    // UPDATE: I misunderstood how to define __device__ variables when I said that
    // but it still applies that this is the end of the process and that it'll be auto-freed
    // although it isn't ideal that the stuff is all allocated until the end of the program's life.

    return 0;
}
