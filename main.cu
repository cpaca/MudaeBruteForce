#include <iostream>
#include <fstream>
#include <string>

std::string* getLines(const std::string& fileName, size_t& arrSize){
    std::ifstream bundleFile;
    bundleFile.open(fileName);
    std::string line;

    size_t maxSize = 5;
    size_t currSize = 0;
    auto* lines = new std::string[maxSize];

    while(std::getline(bundleFile, line)){
        if(currSize == maxSize){
            // need to make lines bigger
            auto* newLines = new std::string[maxSize*2];
            for(size_t i = 0; i < maxSize; i++){
                newLines[i] = lines[i];
            }

            // no longer need lines, so clean up memory
            delete[] lines;
            lines = newLines;

            // and record new max size
            maxSize *= 2;
        }

        lines[currSize] = line;
        currSize++;
    }

    bundleFile.close();

    // return
    arrSize = currSize;
    return lines;
}

/**
 * Gets data about either a bundle or a series from a line.
 * @param line The line to read bundle information about
 * @param name (Return) The name of the bundle or series
 * @return Data about the bundle or series, terminated by -1. (Note that size_t is unsigned.)
 * Data format is designed in the PyCharm processor, but assuming nothing has changed since this was written:
 * Bundle Data Format:
 *  First item is bundle size
 *  Remaining items are the IDs of series in the bundle.
 * Series Data Format:
 *  First item is the series size
 *  Second item is the sizes of the series.
 *  Third item is -1
 */
size_t* getLineData(std::string line, std::string& name){
    name = "";
    size_t maxSize = 5;
    size_t currSize = 0;
    auto* arr = new size_t[maxSize];
    while(!line.empty()){
        auto index = line.find('$');
        std::string token;
        if(index != std::string::npos){
            token = line.substr(0, index);
            line = line.substr(index+1); // drop the $
        }
        else{
            token = line;
            line = "";
        }

        if(name.empty()){
            // token is name
            name = token;
        }
        else{
            // token is an item
            // check if we can fit the item:
            if(currSize == maxSize){
                // need to resize array for new item
                auto* newArr = new size_t[maxSize * 2];
                for(size_t i = 0; i < currSize; i++){
                    newArr[i] = arr[i];
                }

                // then delete the existing array
                delete[] arr;

                // then put the new array info in
                arr = newArr;
                maxSize *= 2;
            }

            // add item
            arr[currSize] = std::stoi(token);

            // item added, update currSize
            currSize++;
        }
    }
    // final array is prepared
    // trim it so it's not massive in memory
    auto* out = new size_t[currSize+1]; // because currSize item will be -1
    // copy over items
    for(size_t i = 0; i < currSize; i++){
        out[i] = arr[i];
    }
    // put the -1 item in
    out[currSize] = -1;
    // we don't need the old version so remove it to save memory
    delete[] arr;
    // and return
    return out;
}

/**
 * Converts a size_t array (in host memory) to a size_t array (in device memory)
 * Also calls delete[] on the host memory version for you. You have to call cudaFree() on your own, though.
 */
void convertArrToCuda(size_t* &arr, size_t arrSize){
    size_t* deviceArr;
    cudaMalloc(&deviceArr, arrSize * sizeof(size_t));
    cudaMemcpy(deviceArr, arr, arrSize * sizeof(size_t), cudaMemcpyHostToDevice);
    delete[] arr;
    arr = deviceArr;
}

/**
 * Copies a size_t array in host memory to a size_t array in device memory. Performs the cudaMalloc for you.
 * Calls delete[] on the host memory for you, though you have to cudaFree on your own.
 * Or let the application exit to clean up the memory.
 * @param to A size_t array in host memory.
 * @param from A size_t array in device memory.
 * @param arrSize The size of each array.
 */
void copyArrToCuda(size_t* &to, size_t* from, size_t arrSize){
    cudaMalloc(&to, arrSize * sizeof(size_t));
    cudaMemcpy(&to, from, arrSize * sizeof(size_t), cudaMemcpyHostToDevice);
    delete[] from;
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

/**
 * Generates a random value, then updates the seed.
 * This is statistical randomness, not cryptographic randomness.
 * Also note that this method simply returns the seed; this is done because it makes some shorthand things easier
 * (i.e. you can do generateRandom(seed)%limit instead of having seed = generateRandom(seed); num = seed%limit)
 * @param seed The "seed" of the randomness.
 * @return
 */
__device__ size_t generateRandom(size_t &seed){
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    // These are the values that newlib uses, and while there is a note saying that not all of the values are ideal
    // this is one of the only ones that uses modulus 2^64
    seed = (6364136223846793005*seed) + 1;
    return seed;
}

/**
 * Int-to-string. (I-to-S)
 * Assumes str is large enough, and just overrides str.
 */
__device__ void deviceItos(char* &str, size_t num){
    if(num == 0){
        str[0] = '0';
        str[1] = NULL;
        return;
    }
    // find the power of 10 bigger than this
    size_t divBy = 1;
    while((num/divBy) != 0){
        divBy *= 10;
    }
    // num/divBy = 0, fix that:
    size_t strIdx = 0;
    while(divBy != 1){
        // remember at first num/divBy = 0 so don't print that 0
        divBy /= 10;
        str[strIdx] = '0' + ((num/divBy)%10); // NOLINT(cppcoreguidelines-narrowing-conversions)
        strIdx++;
    }
    str[strIdx] = NULL;
}

__device__ void deviceStrCat(char* dest, const char* src){
    size_t destIdx = 0;
    while(dest[destIdx]){
        destIdx++;
    }
    size_t srcIdx = 0;
    while(src[srcIdx]){
        dest[destIdx] = src[srcIdx];
        destIdx++;
        srcIdx++;
    }
    dest[destIdx] = NULL;
}

/**
 * Activates a bundle located at bundleIndex.
 * Returns 0 if there is no value gained by activating the bundle.
 * @param bundleIndex The index of the bundle in bundleSeries. Therefore, bundleSeries[bundleIndex] = bundleSize.
 * @param deviceBundles 1D array of bundle data
 * @param deviceSeries 1D array of series data
 * @param activatedSets List of already-activated series
 * @return The value gained from this.
 */
// This parameter list is long and I don't like it...
__device__ size_t activateBundle(const size_t &bundleIndex, const size_t* deviceBundles, const size_t* deviceSeries,
                                 bool* activatedSets){
    size_t improvedValue = 0;
    // there's a lot of error-checking that I'd like to do but can't without having twice as many params.
    size_t bundleOffset = bundleIndex+1; // 0 is size which we don't care about
    while(deviceBundles[bundleOffset] != -1){
        size_t seriesID = deviceBundles[bundleOffset];
        if(!activatedSets[seriesID]){
            // series isn't activated, activate it for this bundle.
            activatedSets[seriesID] = true;
            // and add its value to this
            improvedValue += deviceSeries[(2*seriesID) + 1];
        }
        bundleOffset++; // check next series in bundle.
    }
    return improvedValue;
}

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
// Maximum number of bundles/series that can be activated.
const size_t MAX_DL = 50;
// Maximum number of free bundles.
// Can be changed whenever, but keep it low or CUDA will demand much more memory than necessary.
const size_t MAX_FREE_BUNDLES = 5;
// Overlap limit, defined in Mudae
const size_t OVERLAP_LIMIT = 30000;

// For each bundle, what series are in it?
// (Index 0 is also the bundle's size.)
__device__ size_t* bundleSeries = nullptr;
// Index of each bundle in bundleSeries. So bundleSeries[bundleIndices[n]] is the first index of a bundle in bundleSeries.
__device__ size_t* bundleIndices = nullptr;

/* For each set, what bundles contain it?
 * The format is... kind of a long description.
 * First, let setBundlesSetSize = (numBundles/sizeof(size_t))
 * And for shorthand, let sBSS = setBundlesSetSize
 * Indices setBundles[setNum * sBSS] to setBundles[(setNum+1)*sBSS - 1] are the indices for set setNum
 * In other words, to loop over all values in setBundles relevant to a set:
 * for(int i = 0; i < sBSS; i++){ do something with setBundles[setNum*sBSS + i]}
 *
 * Now express setBundles[0], setBundles[1], ... as a bitstream.
 * The first bit represents if the set is in bundle # 0
 * The second bit represents if the set is in bundle # 1
 * etc.
 * Because this is a bitstream and size_t is 64-bits:
 * the 65th bit (aka, the first bit of setBundles[1], aka setBundles[1]&0) represents if the set is in bundle #65
 *
 * Note that this is setBundles, so it needs to work for all SETS. Even Bundles.
 * Also note that for bundles, their "bitstream" is all 0s except for itself, where it is 1.
 * TODO: Optimization: Bundles which are entirely contained within other bundles (ex cover corp is in vtubers)
 *  could have a better setBundles value.
 */
__device__ size_t* setBundles = nullptr;
__device__ size_t setBundlesSetSize = -1; // note that setBundles[-1] = illegal (unsigned type)

// Data about each series.
// deviceSeries[2n] is the size of series n
// deviceSeries[2n+1] is the value of series n
__device__ size_t* deviceSeries = nullptr;

// Free bundles.
// If freeBundles[n] is non-zero, then bundle n is free.
__device__ size_t* freeBundles = nullptr;

/**
 * Converts bundle data into a 1D array created using cudaMalloc
 * @param bundleData Bundle data. Each element in the array is a bundle, represented by another array terminated by -1.
 *  Note that each bundle need not be the same size. (Hence the -1 termination.)
 * @param numBundles The number of bundles.
 * @param[out] deviceBundles Output: The bundleData, represented as a 1D array. Basically, a flattened version of bundleData.
 * @param[out] deviceIndices The index (in bundleSeries) of each bundle. Note that deviceIndices[0] = 0 since the first
 *  bundle will obviously start at the first index of bundleSeries.
 */
void copyBundlesToDevice(size_t** bundleData, size_t numBundles){
    // First, we need to convert bundleData into a 1D array.
    // In other words, we need to flatten it.
    // First, find out how big the flattened array will be.
    // While we're doing this, we can also find out the indices of each bundle.
    size_t flattenedSize = 0;
    auto* hostIndices = new size_t[numBundles];
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        // The start of this bundle is at index flattenedSize
        hostIndices[bundleNum] = flattenedSize;

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
        bundleIdx++;
    }

    // Now we can construct our 1D array
    auto* hostBundles = new size_t[flattenedSize];
    size_t deviceBundleIdx = 0;
    // Copy each array over to their respective position...
    // Normally I'd just figure out memcpy but, well, no trust.
    // Also I haven't formally been taught memcpy so I don't trust my knowledge of it.
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        if(hostIndices[bundleNum] != deviceBundleIdx){
            // this bundle isn't starting in the right spot.
            throw std::logic_error("A bundle isn't starting in the right spot.");
        }
        size_t bundleIdx = 0;
        while(bundleData[bundleNum][bundleIdx] != -1){
            hostBundles[deviceBundleIdx] = bundleData[bundleNum][bundleIdx];
            deviceBundleIdx++;
            bundleIdx++;
        }
        // and then copy over the -1
        hostBundles[deviceBundleIdx] = -1;
        deviceBundleIdx++;
    }
    if(deviceBundleIdx != flattenedSize){
        throw std::logic_error("The flattened list isn't the right size.");
    }

    // Finally, CUDAify bundleSeries and deviceIndices
    copyArrToCuda(bundleIndices, hostIndices, numBundles);
    copyArrToCuda(bundleSeries, hostBundles, flattenedSize);
}

__device__ size_t getSetSize(size_t setNum, size_t numSeries){
    if(setNum < numSeries){
        return deviceSeries[2*setNum];
    }
    else{
        return bundleSeries[bundleIndices[setNum - numSeries]];
    }
}

// Initializes setBundles.
// Note that this needs to be placed after the fuckton of variables because it manipulates some of them.
void initializeSetBundles(size_t numBundles, size_t numSeries, size_t** bundleData){
    // create setBundles.
    // Explanation of *8: sizeof() returns size in bytes, I want size in bits.
    // Explanation of +1: If there are 7 bundles, setSize should be 1, not 0.
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
                    setBundlesValue = setBundlesValue | (1 << bundleOffset);
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

    convertArrToCuda(host_setBundles, numSets * host_setBundlesSetSize);
    cudaMemcpyToSymbol(setBundles, &host_setBundles, sizeof(host_setBundles));
}

/**
 * Given two lists of bundles in the setBundles format, this returns whether there is an overlap in those bundles.
 * So 0b00010111 and 0b00100010 have a bundle overlap (bundle 6) but neither has a bundle overlap with 0b11000000
 * as neither of them uses bundles 0 or 1.
 */
__device__ bool bundleOverlap(const size_t* A, const size_t* B){
    for(size_t offset = 0; offset < setBundlesSetSize; offset++){
        if(A[offset] & B[offset]){
            return true;
        }
    }
    return false;
}

__global__ void findBest(const size_t numBundles, const size_t numSeries){
    // Set up randomness
    printf("CUDA setting up randomness\n");
    size_t numSets = numSeries + numBundles;
    size_t seed = (blockIdx.x << 10) + threadIdx.x;
    seed = generateRandom(seed) ^ clock();

    // There are three DL limitations.
    printf("CUDA initializing DL limitations 1 and 2\n");
    // You can disable at most a certain number of series. (MAX_DL)
    auto* disabledSets = new size_t[MAX_DL + MAX_FREE_BUNDLES];
    size_t disabledSetsIndex = 0;
    size_t DLSlotsUsed = 0;
    // You can disable at most a certain number of characters. (Overlap limit.)
    size_t remainingOverlap = OVERLAP_LIMIT;
    // A series cannot be disabled twice.
    // That limitation is handled when the score is calculated.

    // Apply the free bundles.
    printf("CUDA applying free bundles.\n");
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        if(freeBundles[bundleNum] != 0){
            disabledSets[disabledSetsIndex] = bundleNum + numSeries;
            disabledSetsIndex++;
        }
    }

    // Create a theoretical DL.
    // This addresses restriction 1.
    printf("CUDA creating theoretical DL\n");
    size_t numFails = 0;
    while(DLSlotsUsed < MAX_DL && numFails < 1000){
        size_t setToAdd = generateRandom(seed) % numSets;
        size_t setSize = getSetSize(setToAdd, numSeries);
        // This addresses restriction 2.
        if(setSize < remainingOverlap){
            // Add this set to the DL
            disabledSets[disabledSetsIndex] = setToAdd;
            disabledSetsIndex++;
            DLSlotsUsed++;
            numFails = 0;
        }
        else{
            numFails++;
        }
    }

    // To address restriction 3, we need to know what bundles are used.
    printf("CUDA calculating used bundles\n");
    auto* bundlesUsed = new size_t[setBundlesSetSize];
    for(size_t item = 0; item < disabledSetsIndex; item++){
        if(item > numSeries){
            size_t bundleNum = item - numSeries;
            size_t bundlesUsedWordSize = 8 * sizeof(size_t);
            size_t bundlesUsedIndex = bundleNum / bundlesUsedWordSize;
            size_t bundleOffset = bundleNum % bundlesUsedWordSize;
            bundlesUsed[bundlesUsedIndex] |= 1 << bundleOffset;
        }
        // else: not a bundle so not apart of bundlesUsed
    }

    // Calculate the score.
    printf("CUDA calculating score\n");
    size_t score = 0;
    for(size_t seriesNum = 0; seriesNum < numSeries; seriesNum++){
        size_t* seriesBundles = setBundles + (setBundlesSetSize * seriesNum);
        bool applySeries = bundleOverlap(bundlesUsed, seriesBundles);
        if(!applySeries){
            // check if the series is in the DL.
            for(size_t DLIdx = 0; DLIdx < disabledSetsIndex; DLIdx++){
                if(disabledSets[DLIdx] == seriesNum){
                    applySeries = true;
                    break; // only breaks out of one for loop
                }
            }
        }

        if(applySeries){
            // Add this series's value to the score.
            score += deviceSeries[(2 * seriesNum) + 1];
        }
    }

    printf("CUDA checking if this is the best score.\n");
    size_t oldBest = atomicMax(&bestScore, score);
    if(oldBest <= score){
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
        deviceStrCat(betterStr, "\n");
        size_t secondCheck = atomicMax(&bestScore, score);
        if(secondCheck <= score) {
            printf("%s", betterStr);
        }
        delete[] num;
    }

    printf("CUDA findBest finished.\n");

    // Free up memory.
    delete[] bundlesUsed;
    delete[] disabledSets;
}

int main() {
    size_t numBundles;
    // first read the bundlesStr
    std::string* bundlesStr = getLines("../bundle_data.txt", numBundles);
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
    std::string* seriesStr = getLines("../series_data.txt", numSeries);
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
    std::string freeBundleNames[] = {"Western", "Hentai"};
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
    copyArrToCuda(freeBundles, host_freeBundles, numBundles);

    initializeSetBundles(numBundles, numSeries, bundleData);

    // time to do CUDA.
    // https://forums.developer.nvidia.com/t/how-to-cudamalloc-two-dimensional-array/4042
    // Mother fucker, I'm gonna have to convert the bundleData into a 1D array.
    // At least I can use seriesData as a 2D array, using the same schema as before.
    // Can't do that for bundleData because bundleData is non-rectangular.
    // ... bleh, cudaMallocPitch is annoying, I might also do seriesData as a 1D array...

    copyBundlesToDevice(bundleData, numBundles);

    deviceSeries = nullptr;
    copySeriesToDevice(seriesData, numSeries, deviceSeries);
    if(deviceSeries == nullptr){
        throw std::logic_error("Device series did not get overwritten properly.");
    }

    // Non-array values are available in Device memory. Proof: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    // Section 3.2.2 uses "int N" in both host and device memory.

    // https://stackoverflow.com/questions/23260074/allocating-malloc-a-double-in-cuda-device-function
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);

    // makeError<<<2, 512>>>(numBundles, numSeries);
    findBest<<<1, 1>>>(numBundles, numSeries);
    cudaDeviceSynchronize();
    cudaError_t lasterror = cudaGetLastError();
    if (lasterror != cudaSuccess) {
        const char *errName = cudaGetErrorName(lasterror);
        printf("%s\n", errName);
//        break;
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
    // <https://forums.developer.nvidia.com/t/will-cuda-free-the-memory-when-my-application-exit/16113/5>
    // and they'll get auto-freed when application exit.

    return 0;
}
