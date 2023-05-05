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
 * Converts bundle data into a 1D array created using cudaMalloc
 * @param bundleData Bundle data. Each element in the array is a bundle, represented by another array terminated by -1.
 *  Note that each bundle need not be the same size. (Hence the -1 termination.)
 * @param numBundles The number of bundles.
 * @param[out] deviceBundles Output: The bundleData, represented as a 1D array. Basically, a flattened version of bundleData.
 * @param[out] bundleIndices The index (in deviceBundles) of each bundle. Note that bundleIndices[0] = 0 since the first
 *  bundle will obviously start at the first index of deviceBundles.
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
        bundleIdx++;
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

    // Finally, CUDAify deviceBundles and bundleIndices
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
 * @param bundleIndex The index of the bundle in deviceBundles. Therefore, deviceBundles[bundleIndex] = bundleSize.
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

// Score to beat.
__device__ size_t bestScore = 0;
// Maximum number of bundles/series that can be activated.
const size_t MAX_DL = 50;
// Overlap limit, defined in Mudae
const size_t OVERLAP_LIMIT = 30000;

__global__ void makeError(const size_t numBundles, const size_t numSeries) {
    // printf("CUDA findBest Executing\n");
    // The first numSeries indices (ie from 0 to numSeries-1) represent if a series is active
    // The rest represent if a bundle is active
    size_t numSets = numSeries + numBundles;
    bool *activatedSets = new bool[numSets];
    // printf("CUDA Initializing activatedSets\n");
    char *debugNum = new char[10];
    for (size_t i = 0; i < numSets; i++) {
        activatedSets[i] = false;
        printf("Moving on\n");
    }
    delete[] debugNum;
}

/**
 * Attempts to find the best Disable List.
 * @param deviceBundles The bundles. Format: [BundleSize, SeriesID, SeriesID, ... SeriesID, -1, BundleSize, ...]
 * @param bundleIndices The index of each BundleSize value.
 * @param numBundles How many bundles there are
 * @param deviceSeries The series. Format: [SeriesSize, SeriesValue, SeriesSize, SeriesValue, ...]
 * @param numSeries How many series there are. Note that deviceSeries is of length numSeries*2
 * @param freeBundles Which bundles are "free" (for example, all western series can be disabled for free)
 */
__global__ void findBest(const size_t* deviceBundles, const size_t* bundleIndices, const size_t numBundles,
                         const size_t* deviceSeries, const size_t numSeries,
                         const size_t* freeBundles){
    // printf("CUDA findBest Executing\n");
    // The first numSeries indices (ie from 0 to numSeries-1) represent if a series is active
    // The rest represent if a bundle is active
    size_t numSets = numSeries + numBundles;
    bool* activatedSets = new bool[numSets];
    // printf("CUDA Initializing activatedSets\n");
//    char* debugNum = new char[10];
    for(size_t i = 0; i < numSets; i++){
//        deviceItos(debugNum, i);
//        deviceStrCat(debugNum, "\n");
//        printf("%s", debugNum);
        activatedSets[i] = false;
    }
//    delete[] debugNum;

    // printf("CUDA Calculating seed\n");
    size_t seed = (blockIdx.x << 10) + threadIdx.x;
    seed = generateRandom(seed) ^ clock();

    // printf("CUDA Creating Limits\n");
    // Maximize score, but...
    size_t score = 0;
    // stay in the overlap limit...
    size_t remainingOverlap = OVERLAP_LIMIT;
    // and only so many items can be disabled.
    size_t DLSlotsUsed = 0;
    auto* disabledSets = new size_t[MAX_DL];

    // Activate the freeBundles
    for(size_t bundleNum = 0; bundleNum < numBundles; bundleNum++){
        if(freeBundles[bundleNum] != 0){
            // This bundle is active.
            // we don't actually need to save the score for this since all sets will have this for free
            // and don't have to check activatedSets because it just got set to all-false
            activatedSets[numSeries + bundleNum] = true;
            activateBundle(bundleIndices[bundleNum], deviceBundles, deviceSeries, activatedSets);
        }
    }

    // printf("CUDA entering while loop\n");
    size_t numFails = 0;
    while(numFails < 100){
        // Check for DL slots
        // printf("CUDA while Loop checking for DL slots\n");
        if(DLSlotsUsed >= MAX_DL){
            break; // can't disable anything else
        }
        // Pick a set to disable.
        // printf("CUDA picking a set to disable\n");
        size_t setToDisable = generateRandom(seed) % numSets;
        if(activatedSets[setToDisable]){
            // Already disabled this one.
            numFails++;
            continue;
        }
        // LOGIC NOTE:
        // If we're here, then one of two things is true:
        // Either we're going to activate the set
        // Or the set is too fat to be activated.
        // If we activate the set:
        //  then it will be noted in activatedSets
        //  and activatedSets[setToDisable] = true
        // If the set is too fat to be activated:
        //  It will always be too fat to be activated
        //  and any bundle the set is in will DEFINITELY be too fat to be activated, anyway
        //  and we should "fail faster" next time
        //  So activatedSets[setToDisable] = true (fail at the above statement)
        activatedSets[setToDisable] = true;
        // End of fun logic

        // See if set is small enough
        // printf("CUDA calculating set size\n");
        size_t setSize;
        if(setToDisable < numSeries){
            // printf("CUDA Calculating series set size\n");
            setSize = deviceSeries[2 * setToDisable];
        }
        else{
            // printf("CUDA Calculating bundle set size\n");
            setSize = deviceBundles[bundleIndices[setToDisable - numSeries]];
        }
        // Is it?
        // printf("CUDA checking if set is too large\n");
        if(setSize > remainingOverlap){
            // Doesn't fit.
            numFails++;
            continue;
        }

        // We're here.
        // We have space in $dl
        // We have overlap limit available.
        // This series isn't already disabled.
        // Activate the set. (Already done by fun logic above)
        // Account for the set's fatness.
        // printf("CUDA applying set\n");
        remainingOverlap -= setSize;
        // Add the set's value.
        if(setToDisable < numSeries){
            // This set is a series, add the series's value
            score += deviceSeries[(2*setSize) + 1];
        }
        else{
            // This set is a BUNDLE.
            size_t bundleValue =
                    activateBundle(setToDisable - numSeries, deviceBundles, deviceSeries, activatedSets);
            if(bundleValue == 0){
                continue; // don't add this bundle to disableSeries
            }
            score += bundleValue;
        }
        // and note that this series is disabled
        // printf("CUDA noting series disability\n");
        disabledSets[DLSlotsUsed] = setToDisable;
        DLSlotsUsed++;
        // reset fails cause why not
        numFails = 0;
    }
    // printf("CUDA exiting While Loop\n");
    size_t oldBest = atomicMax(&bestScore, score);
    if(oldBest <= score){
        char betterStr[1000] = "Better DL Found, score: ";
        char* num = new char[10];
        deviceItos(num, score);
        deviceStrCat(betterStr, num);
        deviceStrCat(betterStr, ", series used, ");
        for(size_t DLIdx = 0; DLIdx < DLSlotsUsed; DLIdx++){
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

    // Free up memory.
    delete[] disabledSets;
    delete[] activatedSets;
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
        size_t idx = 1;
        auto expectedSize = bundleData[i][0];
        size_t actualSize = 0;
        auto seriesID = bundleData[i][idx];
        while(seriesID != -1){
            if(seriesID >= numSeries){
                std::cerr << "Bundle has invalid SeriesID: " << bundleNames[i] << "\n";
                return 1;
            }
            actualSize += seriesData[seriesID][0];
            idx++;
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
    auto* freeBundles = new size_t[numBundles];
    std::string freeBundleNames[] = {"Western", "Hentai"};
    int numFreeBundles = sizeof(freeBundleNames)/sizeof(freeBundleNames[0]);
    int freeBundlesFound = 0;
    for(size_t i = 0; i < numBundles; i++){
        std::string bundleName = bundleNames[i];
        freeBundles[i] = 0;
        for(const std::string& freeBundleName : freeBundleNames){
            if(freeBundleName == bundleName){
                freeBundles[i] = ~0; // not 0 = all 1s (0xfff...)
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
    convertArrToCuda(freeBundles, numBundles);

    // time to do CUDA.
    // https://forums.developer.nvidia.com/t/how-to-cudamalloc-two-dimensional-array/4042
    // Mother fucker, I'm gonna have to convert the bundleData into a 1D array.
    // At least I can use seriesData as a 2D array, using the same schema as before.
    // Can't do that for bundleData because bundleData is non-rectangular.
    // ... bleh, cudaMallocPitch is annoying, I might also do seriesData as a 1D array...

    size_t* deviceBundles = nullptr;
    size_t* bundleIndices = nullptr;
    copyBundlesToDevice(bundleData, numBundles, deviceBundles, bundleIndices);
    if(deviceBundles == nullptr || bundleIndices == nullptr){
        throw std::logic_error("Device bundles and/or bundleIndices did not get overwritten properly.");
    }

    size_t* deviceSeries = nullptr;
    copySeriesToDevice(seriesData, numSeries, deviceSeries);
    if(deviceSeries == nullptr){
        throw std::logic_error("Device series did not get overwritten properly.");
    }

    // Objects available in Device memory:
    //  deviceBundles, bundleIndices
    //  deviceSeries
    //  freeBundles
    // Non-array values are available in Device memory. Proof: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    // Section 3.2.2 uses "int N" in both host and device memory.

    // https://stackoverflow.com/questions/23260074/allocating-malloc-a-double-in-cuda-device-function
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);

    // makeError<<<2, 512>>>(numBundles, numSeries);
    for(int i = 0; i < 128; i++) {
        findBest<<<64, 1024>>>(deviceBundles, bundleIndices, numBundles, deviceSeries, numSeries, freeBundles);
        cudaDeviceSynchronize();
        cudaError_t lasterror = cudaGetLastError();
        if (lasterror != cudaSuccess) {
            const char *errName = cudaGetErrorName(cudaGetLastError());
            printf("%s\n", errName);
            break;
        }
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
    cudaFree(freeBundles);
    cudaFree(deviceSeries);
    cudaFree(deviceBundles);
    cudaFree(bundleIndices);

    return 0;
}
