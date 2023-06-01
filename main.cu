#include <iostream>
#include <string>

#include "strUtils.cu"
#include "hostDeviceUtils.cu"
#include "randUtils.cu"
#include "types.cu"

// Maximum number of bundles/series that can be activated.
#define MAX_DL 50
// Maximum number of free bundles.
// Can be changed whenever, but keep it low or CUDA will demand much more memory than necessary.
#define MAX_FREE_BUNDLES 5
// Overlap limit, defined in Mudae
#define OVERLAP_LIMIT 30000
// How many blocks to run.
// Note that each block gets 512 threads.
#define NUM_BLOCKS (1 << 12)
// "MinSize" is a variable determining the minimum size a series needs to be to be added to the DL.
// MinSize gets divided by 2 while the remainingOverlap exceeds minSize, so even a minSize of 2^31 will get fixed
// down to remainingOverlap levels.
// MAX_MINSIZE determines the maximum value minSize can be.
#define MAX_MINSIZE 100
// Whether or not to run the in-code Profiler.
// Note that the profiler is implemented in code, not using an actual profiler
// like nvcc or nvvp
#define PROFILE true

bool bundleContainsSet(size_t setNum, size_t bundleNum, size_t numBundles, size_t numSeries, size_t** bundleData){
    if(bundleNum >= numBundles){
        return false;
    }
    if(setNum >= numSeries){
        // this is a bundle, not a series
        setNum -= numSeries;
        return setNum == bundleNum;
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
// TODO: Optimization: Bundles which are entirely contained within other bundles (ex cover corp is in vtubers)
//  could have a better setBundles value.
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

// Initializes setBundles.
// Note that this needs to be placed after the fuckton of variables because it manipulates some of them.
void initializeSetBundles(size_t numBundles, size_t numSeries, size_t** bundleData){
    // create setBundles.
    // Explanation of *8: sizeof() returns size in bytes, I want size in bits.
    // Explanation of +1: If there are 7 bundles, setSize_t should be 1, not 0.
    // host_ added because I can't write device data directly @ host level.
    const size_t host_setBundlesSetSize = (numBundles / (sizeof(size_t) * 8)) + 1;
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
                size_t bundleNumToCheck = (sizeof(size_t) * i) + bundleOffset;
                if(bundleContainsSet(setNum, bundleNumToCheck, numBundles, numSeries, bundleData)){
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
            size_t setBundlesVal = setBundles[(host_setBundlesSetSize * setNum) + i];
            while(setBundlesVal > 0){
                if(setBundlesVal%2 != 0) {
                    std::cout << std::to_string(bundleNum) << "\n";
                }
                setBundlesVal /= 2;
                bundleNum++;
            }
        }
        std::cout << "\n";
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
        if(seriesValue == 0){
            seriesSize = OVERLAP_LIMIT+1;
        }
        if(seriesSize > ((size_t) get_unsigned_max<setSize_t>())){
            std::cout << "One of the sets is too fat for setSize_t: " << std::to_string(seriesNum) << "\n";
            return;
        }
        if(seriesSize > OVERLAP_LIMIT){
            seriesSize = OVERLAP_LIMIT+1;
        }

        host_setSizes[seriesNum] = seriesSize;
    }
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        size_t* bundlePtr = bundleData[bundleNum];
        host_setSizes[numSeries + bundleNum] = bundlePtr[0];
        if(host_freeBundles[bundleNum] != 0){
            // Set this bundle's series's to OVERLAP_LIMIT+1
            bundlePtr++;
            while(true){
                size_t seriesNum = *bundlePtr;
                if(seriesNum == -1){
                    break;
                }
                host_setSizes[seriesNum] = OVERLAP_LIMIT+1;
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

/**
 * If set represents the Set ID of a bundle, then bundlesUsed is modified to acknowledge that that bundle is
 * activated.
 * @param numSeries The total number of series there are.
 * @param bundlesUsed MAY BE MODIFIED to acknowledge this set being added to bundlesUsed.
 * @param setToAdd The Set ID of a bundle.
 */
__device__ void activateBundle(const size_t numSeries, size_t *bundlesUsed, size_t set) {
    if(set >= numSeries){
        // setToAdd is actually a bundle to add
        // If this bundle is being used, we need to acknowledge that in bundlesUsed
        size_t bundleNum = set - numSeries;
        size_t bundlesUsedWordSize = 8 * sizeof(size_t);
        size_t bundlesUsedIndex = bundleNum / bundlesUsedWordSize;
        size_t bundleOffset = bundleNum % bundlesUsedWordSize;
        bundlesUsed[bundlesUsedIndex] |= (((size_t)1) << bundleOffset);
    }
}

__global__ void findBest(const size_t numBundles, const size_t numSeries){
#if PROFILE
    size_t lastTime = clock64();
    size_t currTime; // set this later when comparing
#endif

    // Set up randomness
    // printf("CUDA setting up randomness\n");
    size_t seed = (blockIdx.x << 10) + threadIdx.x;
    seed = generateRandom(seed) ^ clock();
    // seed = 0; // debug to get the same result every time

    // There are three DL limitations.
    // printf("CUDA initializing DL limitations 1 and 2\n");
    // You can disable at most a certain number of series. (MAX_DL)
    auto* disabledSets = new size_t[MAX_DL + MAX_FREE_BUNDLES];
    size_t disabledSetsIndex = 0;

    // By having DLSlotsUsed start at 1
    // this is the same as "reserving" the last slot for any series
    // This way, when there's one slot left, we can run the Greedy Algorithm to find what the best series would be.
    size_t DLSlotsUsed = 0;

    // You can disable at most a certain number of characters. (Overlap limit.)
    size_t remainingOverlap = OVERLAP_LIMIT;
    // A series cannot be disabled twice.
    // That limitation is handled when the score is calculated.

    // To address restriction 3, we need to know what bundles are used.
    auto* bundlesUsed = new size_t[setBundlesSetSize];
    memset(bundlesUsed, 0, sizeof(size_t) * setBundlesSetSize);

    // Apply the free bundles.
    // printf("CUDA applying free bundles.\n");
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        if(freeBundles[bundleNum] != 0){
            // Add bundle to disabledSets...
            disabledSets[disabledSetsIndex] = bundleNum + numSeries;
            disabledSetsIndex++;

            // ... but more importantly add bundle to bundlesUsed
            // I think the compiler will inline it and apply 0, bundlesUsed, bundleNum
            // so I don't have to.
            activateBundle(numSeries, bundlesUsed, bundleNum + numSeries);
        }
    }

    // IDEA: What if we have a min_size value to not reserve a ton of 1-size seriess.
    size_t origMinSize = generateRandom(seed, MAX_MINSIZE);
    size_t minSize = origMinSize;

    // Create a theoretical DL.
    // This addresses restriction 1.
    // printf("CUDA creating theoretical DL\n");
    size_t numFails = 0;
    // Did some testing and analysis.
    // Due to the way CUDA works, it's best to assume this takes 10k loops to complete.
    // I typically saw it complete in 4-8k loops, but if one thread in a warp needs 10k loops to complete
    // then the other 31 threads will wait 10k loops
    // giving them an effective time of 10k loops

#if PROFILE
    currTime = clock64();
    devicePrintStrNum("Shared memory setup time: ", currTime - lastTime);
    lastTime = currTime;
#endif

    size_t numSets = numSeries + numBundles;
    size_t setSizeToRead = threadIdx.x;
    while(setSizeToRead < numSets){
        bool debug = setSizeToRead == 10932 || setSizeToRead == 10933;
        size_t setSize;
        if(setSizeToRead < numSeries){
            // set is a series
            setSize = deviceSeries[2*setSizeToRead];
            size_t setValue = deviceSeries[(2*setSizeToRead)+1];
            if(setValue == 0){
                setSize = OVERLAP_LIMIT+1;
            }
        }
        else{
            setSize = bundleSeries[bundleIndices[setSizeToRead - numSeries]];
        }

        if(setSize > ((size_t) get_unsigned_max<setSize_t>)){
            devicePrintStrNum("ERROR: One of the sets is too fat for setSize_t: ", setSizeToRead);
            return;
        }

        if(setSize > OVERLAP_LIMIT){
            // I don't think this is possible, since even the basic overlap limit is 20k.
            // But, just in case.
            // -> Note that any catches of this are most likely from the setValue = 0 case
            setSize = OVERLAP_LIMIT+1;
        }

        // I'm really running out of variable names.
        size_t* selfBundles = setBundles + (setBundlesSetSize * setSizeToRead);
        if(bundleOverlap(selfBundles, bundlesUsed)){
            // AKA there's overlap between this and the freeBundles
            setSize = OVERLAP_LIMIT+1;
        }

        setSizes[setSizeToRead] = setSize;
        if(setSizes[setSizeToRead] != global_setSizes[setSizeToRead]){
            devicePrintStrNum("Global setsize disagrees with shared setsize on set ", setSizeToRead);
            devicePrintStrNum("Global: ", global_setSizes[setSizeToRead]);
            devicePrintStrNum("Local: ", setSizes[setSizeToRead]);
        }
        setSizeToRead += blockDim.x;
    }

    __syncthreads();
    printf("Done\n");

#if PROFILE
    currTime = clock64();
    devicePrintStrNum("Profiler: Shared memory calculation time: ", currTime - lastTime);
    lastTime = currTime;
#endif

#if PROFILE
    // Variables used to measure time spent in the while-loop
    size_t lastLoopTime;
    size_t currLoopTime;

    size_t numLoops = 0;

    // Time to do the while loop's comparison operator.
    // Specifically, the DLSLotsUsed < MAX_DL
    size_t whileLoopCompareTime = 0;
    size_t pickSetTime = 0;
    size_t setSizeTime = 0;
    size_t bundleOverlapTime = 0;
    size_t addSetTime = 0;

    lastLoopTime = clock64();
#endif
    while(DLSlotsUsed < MAX_DL && numFails < 1000){
#if PROFILE
        currLoopTime = clock64();
        whileLoopCompareTime += currLoopTime - lastLoopTime;
        lastLoopTime = currLoopTime;
        numLoops++;
#endif
        numFails++;
        size_t setToAdd = generateRandom(seed, numSets);
        // Calculate the size of this set.
        size_t setSize = setSizes[setToAdd];
#if PROFILE
        currLoopTime = clock64();
        pickSetTime += currLoopTime - lastLoopTime;
        lastLoopTime = currLoopTime;
#endif

        if(setSize < minSize){
#if PROFILE
            currLoopTime = clock64();
            setSizeTime += currLoopTime - lastLoopTime;
            lastLoopTime = currLoopTime;
#endif
            continue;
        }

        // This addresses restriction 2.
        if (setSize >= remainingOverlap) {
#if PROFILE
            currLoopTime = clock64();
            setSizeTime += currLoopTime - lastLoopTime;
            lastLoopTime = currLoopTime;
#endif
            continue;
        }
#if PROFILE
        currLoopTime = clock64();
        setSizeTime += currLoopTime - lastLoopTime;
        lastLoopTime = currLoopTime;
#endif

        // Determine redundancy.
        // First, determine the bundles for this set:
        size_t* selfBundles = setBundles + (setBundlesSetSize * setToAdd);
        // Then determine if this set is redundant:
        if(bundleOverlap(selfBundles, bundlesUsed)){
#if PROFILE
            currLoopTime = clock64();
            bundleOverlapTime += currLoopTime - lastLoopTime;
            lastLoopTime = currLoopTime;
#endif
            // This set has already been addressed by a previous bundle.
            // In other words, this set is redundant.
            continue;
        }
        // Note that "don't add a set twice" is another form of redundancy.
        // However, also note that this is, computationally, quite slow.
        bool continueLoop = false;
        for(size_t DLIdx = 0; DLIdx < disabledSetsIndex; DLIdx++){
            size_t setNum = disabledSets[DLIdx];
            if(setToAdd == setNum){
                continueLoop = true;
                break;
            }
        }
        if(continueLoop){
            continue;
        }
#if PROFILE
        currLoopTime = clock64();
        bundleOverlapTime += currLoopTime - lastLoopTime;
        lastLoopTime = currLoopTime;
#endif
        // ADD THIS SET TO THE DL
        disabledSets[disabledSetsIndex] = setToAdd;
        disabledSetsIndex++;
        DLSlotsUsed++;
        remainingOverlap -= setSize;
        numFails = 0;

        activateBundle(numSeries, bundlesUsed, setToAdd);

        while (minSize > remainingOverlap) {
            // I accepted the infinite loop before and it finished really quickly
            // but now I've decided I want to continue collecting the very, very small series
            // just in case they have useful information.
            minSize >>= 1;
        }
#if PROFILE
        currLoopTime = clock64();
        addSetTime += currLoopTime - lastLoopTime;
        lastLoopTime = currLoopTime;
#endif
    }
#if PROFILE
    devicePrintStrNum("While loop execs: ", numLoops);
    devicePrintStrNum(" Time to do while loop compare (DLSLotsUsed < MAX_DL && numFails < 1000): ", whileLoopCompareTime);
    devicePrintStrNum(" Time to pick a set: ", pickSetTime);
    devicePrintStrNum(" Time to validate set's size: ", pickSetTime);
    devicePrintStrNum(" Time to calculate bundle overlap: ", bundleOverlapTime);
    devicePrintStrNum(" Time to add a set to the DL: ", addSetTime);
#endif

#if PROFILE
    currTime = clock64();
    devicePrintStrNum("Profiler: While loop time: ", currTime - lastTime);
    lastTime = currTime;
#endif

    // Calculate the score.
    // printf("CUDA calculating score\n");
    size_t score = 0;

    // Calculate score from bundles
    for(size_t seriesNum = 0; seriesNum < numSeries; seriesNum++){
        size_t* seriesBundles = setBundles + (setBundlesSetSize * seriesNum);
        size_t seriesValue = deviceSeries[(2 * seriesNum) + 1];
        bool overlapsWithBundle = bundleOverlap(bundlesUsed, seriesBundles);
        if(overlapsWithBundle){
            // Add this series's value to the score.
            score += seriesValue;
        }
    }

#if PROFILE
    currTime = clock64();
    devicePrintStrNum("Profiler: Bundle-score calculation time: ", currTime - lastTime);
    lastTime = currTime;
#endif

    // Calculate score directly from series
    for(size_t DLIdx = 0; DLIdx < disabledSetsIndex; DLIdx++){
        size_t setNum = disabledSets[DLIdx];

        // Note that if setNum is a bundle, it'll get caught by bundleOverlap anyway.

        size_t* seriesBundles = setBundles + (setBundlesSetSize * setNum);
        if(bundleOverlap(bundlesUsed, seriesBundles)){
            // Already got covered by the bundles earlier.
            continue;
        }

        // Add this series's score.
        size_t seriesValue = deviceSeries[(2 * setNum) + 1];
        score += seriesValue;
    }

#if PROFILE
    currTime = clock64();
    devicePrintStrNum("Profiler: Series-score calculation time: ", currTime - lastTime);
    lastTime = currTime;
#endif

    // printf("CUDA checking if this is the best score.\n");
    size_t oldBest = atomicMax(&bestScore, score);
    // if this was <= instead of < and the "best score" got achieved, it would spam out that best score nonstop
    if(oldBest < score){
        // Copied straight from the old implementation of findBest.
        char betterStr[1000] = "Better DL Found, score: ";
        char* num = new char[10];
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

        deviceStrCat(betterStr, "\nOriginal minSize: ");
        deviceItos(num, origMinSize);
        deviceStrCat(betterStr, num);

        deviceStrCat(betterStr, "\n\n");

        size_t secondCheck = atomicMax(&bestScore, score);
        // If this was < instead of <=, this would never print because bestScore either = score, from this, or > score,
        // from another thread
        if(secondCheck <= score) {
            printf("%s", betterStr);
        }
        delete[] num;
    }

    // printf("CUDA findBest finished.\n");

    // Free up memory.
    delete[] bundlesUsed;
    delete[] disabledSets;
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
    convertArrToCuda(host_freeBundles, numBundles);
    if(host_freeBundles == nullptr){
        std::cout << "FreeBundles not initialized correctly.";
    }
    cudaMemcpyToSymbol(freeBundles, &host_freeBundles, sizeof(host_freeBundles));

    initializeSetBundles(numBundles, numSeries, bundleData);

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
#if PROFILE
    // Profiling is too hard to read if there's 2 million threads running, all printing profiler info.
    findBest<<<1, 1, sharedMemoryNeeded>>>(numBundles, numSeries);
#else
    std::cout << "Executing FindBest with " << std::to_string(NUM_BLOCKS) << " blocks of 512 threads each.\n";
    findBest<<<NUM_BLOCKS, 512, sharedMemoryNeeded>>>(numBundles, numSeries);
#endif
    cudaDeviceSynchronize();
    clock_t endTime = clock();
    std::cout << "Time taken (seconds): " << std::to_string((endTime - startTime)/(double)CLOCKS_PER_SEC) << "\n";

    cudaError_t lasterror = cudaGetLastError();
    if (lasterror != cudaSuccess) {
        const char *errName = cudaGetErrorName(lasterror);
        printf("%s\n", errName);
    }
    else{
        printf("No CUDA errors.\n");
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
