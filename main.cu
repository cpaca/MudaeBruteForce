#include <iostream>
#include <string>

#include "constVars.cu"
#include "globalVars.cu"
#include "strUtils.cu"
#include "hostDeviceUtils.cu"
#include "randUtils.cu"
#include "types.cu"
#include "profileUtils.cu"
#include "taskManager.cu"
#include "task.cu"

bool bundleContainsSet(size_t setNum,
                       size_t bundleNum,
                       size_t numBundles,
                       size_t numSeries,
                       size_t **bundleData,
                       size_t **seriesData) {
    if(bundleNum >= numBundles){
        return false;
    }
    if(setNum >= numSeries){
        // this is a bundle, not a series
        setNum -= numSeries;
        if(setNum == bundleNum){
            return true;
        }
        // Now we check if bundleNum contains setNum.
        // If it is, then bundleNum is the "big bundle" and setNum is the "small bundle"
        size_t* bigBundlePtr = bundleData[bundleNum];
        size_t* smallBundlePtr = bundleData[setNum];
        // we could do a size check but honestly this is on the CPU, I don't care about performance *that* much
        bigBundlePtr++;
        smallBundlePtr++;

        // 100% exploiting the fact that series_data and bundle_data are in numerical order.
        while(true){
            size_t bigBundleSeries = *bigBundlePtr;
            size_t smallBundleSeries = *smallBundlePtr;
            if(smallBundleSeries == -1) {
                // end of the small bundle!
                return true;
            }
            else if(bigBundleSeries < smallBundleSeries){
                // big pointer needs to keep going until it finds smallBundleSeries
                bigBundlePtr++;
            }
            else if(bigBundleSeries == smallBundleSeries){
                // shared series
                bigBundlePtr++;
                smallBundlePtr++;
            }
            else{
                // small bundle contains a set that big bundle doesn't!
                size_t* smallBundleSetPtr = seriesData[smallBundleSeries];
                // size_t smallSetSize = smallBundleSetPtr[0]; // unused
                size_t smallSetValue = smallBundleSetPtr[1];
                if(smallSetValue == 0){
                    // Don't care that it's not a sub-bundle
                    // since it won't affect the score either way
                    bigBundlePtr++;
                    smallBundlePtr++;
                }
                else{
                    // Small bundle contains an *important* set the big bundle doesn't!
                    return false;
                }
            }
        }
    }
    // This is a series and a valid bundle number.
    // bundlePtr points to the start of the bundle.
    // was done this way before because I was using bundleIndices cause I forgot this isn't device code.
    size_t* bundlePtr = bundleData[bundleNum];
    // Note that the start of the bundle is the SIZE of the bundle, but we actually want the next value.
    bundlePtr++;
    // Search for our series (ie search for setNum)
    // Note that with the optimization on Python's side, bundle_data is in sorted order.
    while((*bundlePtr) < setNum){
        bundlePtr++;
    }
    // Now either bundlePtr is at setNum
    // or bundlePtr is past setNum
    return (*bundlePtr) == setNum;
}

// Initializes setBundles.
// Note that this needs to be placed after the fuckton of variables because it manipulates some of them.
void initializeSetBundles(size_t numBundles, size_t numSeries, size_t** bundleData, size_t** seriesData){
    // create setBundles.
    cudaMemcpyToSymbol(setBundlesSetSize, &host_setBundlesSetSize, sizeof(size_t));
    size_t numSets = numBundles + numSeries;

    // if setBundlesSetSize is -1 for some reason this will allocate petabytes of memory (won't work)
    // or, more likely, throw an error
    auto* host_setBundles = new size_t[numSets * host_setBundlesSetSize];
    // Set up the series.
    // there's a more computationally efficient way to do this but the GPU part is where i'm worried about that.
    for(size_t setNum = 0; setNum < numSets; setNum++){
        for(size_t i = 0; i < host_setBundlesSetSize; i++){
            size_t setBundlesIdx = (setNum * host_setBundlesSetSize) + i;
            size_t setBundlesValue = 0;
            // again *8 because 8 bits in a byte
            for(size_t bundleOffset = 0; bundleOffset < (sizeof(size_t)*8); bundleOffset++){
                size_t bundleNumToCheck = (sizeof(size_t) * 8 * i) + bundleOffset;
                if(bundleContainsSet(setNum, bundleNumToCheck, numBundles, numSeries, bundleData, seriesData)){
                    setBundlesValue = setBundlesValue | (((size_t)1) << bundleOffset);
                }
            }
            host_setBundles[setBundlesIdx] = setBundlesValue;
        }
    }

    // Validation.
    // Comment out if you don't want that validation.
    /*
    for(size_t setNum = 0; setNum < numSets; setNum++){
        std::cout << "Bundles for set " << std::to_string(setNum) << "\n";
        size_t bundleNum = 0;
        for(size_t i = 0; i < host_setBundlesSetSize; i++){
            size_t setBundlesVal = host_setBundles[(host_setBundlesSetSize * setNum) + i];
            std::cout << "i: " << std::to_string(i) << "\n";
            std::cout << "setBundlesVal: " << std::to_string(setBundlesVal) << std::endl;
            for(size_t j = 0; j < (sizeof(size_t) * 8); j++){
                if(setBundlesVal%2 != 0) {
                    std::cout << std::to_string(bundleNum) << "\n";
                }
                setBundlesVal /= 2;
                bundleNum++;
            }
        }
        std::cout << std::endl;
    }
    //*/

    cudaMemcpyToSymbol(setBundlesSetSize, &host_setBundlesSetSize, sizeof(host_setBundlesSetSize));
    convertArrToCuda(host_setBundles, numSets * host_setBundlesSetSize);
    cudaMemcpyToSymbol(setBundles, &host_setBundles, sizeof(host_setBundles));
}

void initializeGlobalSetSizes(size_t numSeries, size_t numBundles, size_t** seriesData, size_t** bundleData,
                              const size_t* host_freeBundles){
    size_t numSets = numSeries + numBundles;
    auto* host_setSizes = new setSize_t[numSets];

    for(size_t seriesNum = 0; seriesNum < numSeries; seriesNum++){
        size_t seriesSize = seriesData[seriesNum][0];
        size_t seriesValue = seriesData[seriesNum][1];
        if(seriesSize > ((size_t) get_unsigned_max<setSize_t>())){
            std::cout << "One of the sets is too fat for setSize_t: " << std::to_string(seriesNum) << "\n";
            return;
        }

        // Observe that in both cases it's not worth checking if the series's value can be added
        if(seriesValue == 0 || (seriesSize > OVERLAP_LIMIT)){
            seriesSize = OVERLAP_LIMIT+1;
        }

        host_setSizes[seriesNum] = seriesSize;
    }
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        size_t* bundlePtr = bundleData[bundleNum];
        host_setSizes[numSeries + bundleNum] = bundlePtr[0];
        if(host_freeBundles[bundleNum] != 0){
            // Set this bundle to OVERLAP_LIMIT+1 cause it's free
            host_setSizes[numSeries + bundleNum] = OVERLAP_LIMIT+1;
            // Set this bundle's series's to OVERLAP_LIMIT+1
            bundlePtr++;
            while(true){
                size_t seriesNum = *bundlePtr;
                if(seriesNum == -1){
                    break;
                }
                if(host_setSizes[seriesNum] <= OVERLAP_LIMIT){
                    // If it's > OVERLAP_LIMIT then see the note from before (not worth looking at)
                    // but if it's < OVERLAP_LIMIT and it's in a free bundle then it's not worth checking
                    // if it's in bundleOverlaps, because it's definitely in bundleOverlaps.
                    host_setSizes[seriesNum] = OVERLAP_LIMIT+2;
                }
                bundlePtr++;
            }
        }
    }

    convertArrToCuda(host_setSizes, numSets * sizeof(setSize_t));
    cudaMemcpyToSymbol(global_setSizes, &host_setSizes, sizeof(host_setSizes));
}

/**
 * Given two lists of bundles in the setBundles format, this returns whether there is an overlap in those bundles.
 * So 0b00010111 and 0b00100010 have a bundle overlap (bundle 6) but neither has a bundle overlap with 0b11000000
 * as neither of them uses bundles 0 or 1.
 */
__device__ bool bundleOverlap(const size_t* A, const size_t* B){
    for(size_t offset = 0; offset < setBundlesSetSize; offset++){
        if((A[offset] & B[offset]) != 0){
            return true;
        }
    }
    return false;
}

__device__ void printDL(Task* task) {
    size_t* disabledSets = task->disabledSets;
    size_t disabledSetsIndex = task->disabledSetsIndex;
    size_t score = task->score;
    size_t remainingOverlap = task->remainingOverlap;

    size_t oldBest = atomicMax(&bestScore, score);
    // if this was <= instead of < and the "best score" got achieved, it would spam out that best score nonstop
    if(oldBest < score){
        // Copied straight from the old implementation of findBest.
        char* betterStr = new char[1000];
        betterStr[0] = '\0';
        char* num = new char[10];

        deviceStrCat(betterStr, "Better DL Found, score: ");
        deviceItos(num, score);
        deviceStrCat(betterStr, num);
        deviceStrCat(betterStr, ", series used, ");
        for(size_t DLIdx = 0; DLIdx < disabledSetsIndex; DLIdx++){
            deviceItos(num, disabledSets[DLIdx]);
            deviceStrCat(betterStr, num);
            deviceStrCat(betterStr, ", ");
        }
        deviceStrCat(betterStr, "\nRemaining overlap: ");
        deviceItos(num, remainingOverlap);
        deviceStrCat(betterStr, num);

        deviceStrCat(betterStr, "\nWriteIdx: ");
        deviceItos(num, liveTaskQueue.writeIdx);
        deviceStrCat(betterStr, num);

        deviceStrCat(betterStr, "\n\n");

        size_t secondCheck = atomicMax(&bestScore, score);
        // If this was < instead of <=, this would never print because bestScore either = score, from this, or > score,
        // from another thread
        if(secondCheck <= score) {
            printf("%s", betterStr);
            // sleep for 10 ms to MAKE SURE the output works
            // had a weird bug where sometimes it wouldnt output
            __nanosleep(10000000);
        }
        delete[] num;
        delete[] betterStr;
    }
}

__device__ size_t getSetSize(const size_t &numSeries, const size_t &setNum){
    if(setNum < numSeries){
        return deviceSeries[2 * setNum];
    }
    else{
        size_t bundleNum = setNum - numSeries;
        return bundleSeries[bundleIndices[bundleNum]];
    }
}

/**
 * Activates a series for a given task.
 * This does NOT read or write to remainingOverlap
 * This increments Score if necessary
 * This assumes seriesNum is a series and NOT A BUNDLE
 * This reads from (does not write to) task->bundlesUsed
 * If a bundle for this series has already been used, this does nothing.
 */
__device__ void activateSeries(Task* task, size_t seriesNum){
    size_t* seriesBundles = setBundles + (setBundlesSetSize * seriesNum);
    size_t* taskBundles = task->bundlesUsed;
    if(bundleOverlap(taskBundles, seriesBundles)){
        // This series has already been added to the Task.
        return;
    }

    size_t seriesValue = deviceSeries[(2*seriesNum) + 1];
    task->score += seriesValue;
}

__global__ void newFindBest(const size_t numBundles, const size_t numSeries){
    // size_t numSets = numBundles + numSeries;
    size_t* clocks = initProfiling();
    while(true){
        // Use this so the program stops and you can profile shit
        if(liveTaskQueue.writeIdx > (1 << 18)){
            break;
        }
        startClock(clocks, 0);
        Task* task = getTask(liveTaskQueue);
        checkpoint(clocks, 0, &getTaskCheckpoint);
        if(task == nullptr){
            continue;
        }
        if(task->DLSlotsRemn <= 0){
            // Nope! Stop. Done. Nothing to do on this task.
            killTask(task);
            continue;
        }
        checkpoint(clocks, 0, &validTaskCheckpoint);

        Task* newTask = copyTask(task);
        checkpoint(clocks, 0, &copyTaskCheckpoint);

        // Delete the setDeleteIndex on task, leave it alone on newTask
        size_t setToDelete = setDeleteOrder[task->setDeleteIndex];
        task->disabledSets[task->disabledSetsIndex] = setToDelete;
        task->disabledSetsIndex++;
        task->DLSlotsRemn--;

        size_t setSize = getSetSize(numSeries, setToDelete);
        checkpoint(clocks, 0, &makeNewTaskCheckpoint);
        if(setSize > task->remainingOverlap){
            killTask(task);
            task = nullptr;
            checkpoint(clocks, 0, &fullTaskCheckpoint);
        }
        else {
            task->remainingOverlap -= setSize;
            if (setToDelete < numSeries) {
                activateSeries(task, setToDelete);
            } else {
                size_t bundleToDelete = setToDelete - numSeries;

                size_t* bundlePtr = bundleSeries + bundleIndices[bundleToDelete];
                bundlePtr++; // Get past the size value...
                checkpoint(clocks, 0, &bundlePtrCheckpoint);
                while((*bundlePtr) != -1){
                    activateSeries(task, *bundlePtr);
                    bundlePtr++;
                }
                activateBundle(numSeries, task, setToDelete);
                checkpoint(clocks, 0, &activateBundleCheckpoint);
            }
        }
        checkpoint(clocks, 0, &deleteSetCheckpoint);

        // Is the new DL good?
        if(task != nullptr) {
            printDL(task);
        }
        // newTask is unchanged so no print

        // Increment setDeleteIndex on both tasks...
        // (Compiler probably optimizes this to the very front for like 1 or 2 machine operations faster)
        if(task != nullptr) {
            task->setDeleteIndex++;
        }
        newTask->setDeleteIndex++;

        // And put both tasks to the front.
        putTask(liveTaskQueue, task);
        putTask(liveTaskQueue, newTask);
        checkpoint(clocks, 0, &finishLoopCheckpoint);
    }
    destructProfiling(clocks);
}

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

    // Non-array values are available in Device memory. Proof: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    // Section 3.2.2 uses "int N" in both host and device memory.

    // https://stackoverflow.com/questions/23260074/allocating-malloc-a-double-in-cuda-device-function
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);

    size_t sharedMemoryNeeded = (numBundles + numSeries) * sizeof(setSize_t);

    // makeError<<<2, 512>>>(numBundles, numSeries);
    clock_t startTime = clock();

    // std::cout << "Executing FindBest with " << std::to_string(NUM_BLOCKS) << " blocks of 512 threads each.\n";
    // findBest<<<NUM_BLOCKS, 512, sharedMemoryNeeded>>>(numBundles, numSeries);
    std::cout << "Shared memory needed: " << std::to_string(sharedMemoryNeeded) << "\n";
    // reminder to self: 40 blocks of 512 threads each
    // for some reason 1024 threads per block throws some sort of error
    newFindBest<<<40, 512, sharedMemoryNeeded>>>(numBundles, numSeries);
    cudaError_t syncError = cudaDeviceSynchronize();

    clock_t endTime = clock();
    printProfilingData();
    std::cout << "Time taken (seconds): " << std::to_string((endTime - startTime)/(double)CLOCKS_PER_SEC) << "\n";

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
