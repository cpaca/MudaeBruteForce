#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error_handler.cu"
#include "read_sets.cuh"
#include <string>
#include <iostream>

int main() {
    std::cout << "Testing IntelliSense\n";

    readFile();

    CUDAErrorCheck(cudaErrorMemoryAllocation);

    size_t numBundles;
    // first read the bundlesStr
    //std::string* bundlesStr = getLines("../working-data/bundle_data.txt", numBundles);
    //auto** bundleData = new size_t*[numBundles];
    //auto* bundleNames = new std::string[numBundles];
    //for(size_t i = 0; i < numBundles; i++){
    //    bundleData[i] = getLineData(bundlesStr[i], bundleNames[i]);
    //}
    //delete[] bundlesStr;

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
    //size_t numSeries;
    //std::string* seriesStr = getLines("../working-data/series_data.txt", numSeries);
    //auto** seriesData = new size_t*[numSeries];
    //auto* seriesNames = new std::string[numSeries];
    //for(size_t i = 0; i < numSeries; i++){
    //    seriesData[i] = getLineData(seriesStr[i], seriesNames[i]);
    //}
    //delete[] seriesStr;

    // validate that no bundlesStr are exceeding seriesID:
    // Validate bundles.
    //for(size_t i = 0; i < numBundles; i++){
    //    auto expectedSize = bundleData[i][0];

    //    size_t idx = 1; // idx 0 is size so not a series
    //    size_t actualSize = 0;
    //    size_t lastSeriesId = 0;
    //    auto seriesID = bundleData[i][idx];
    //    while(seriesID != -1){
    //        if(seriesID >= numSeries){
    //            std::cerr << "Bundle has invalid SeriesID: " << bundleNames[i] << "\n";
    //            return 1;
    //        }
    //        if(seriesID < lastSeriesId){
    //            std::cerr << "Bundle series are not in sorted order: " << bundleNames[i] << "\n";
    //            return 3;
    //        }
    //        actualSize += seriesData[seriesID][0];
    //        idx++;
    //        lastSeriesId = seriesID;
    //        seriesID = bundleData[i][idx];
    //    }

    //    if(expectedSize != actualSize){
    //        std::cerr << "Bundle size is not equivalent to the sum of the sizes of its series.\n";
    //        std::cerr << "Violating BundleID: " << std::to_string(i);
    //        std::cerr << "Violating bundleName: " << bundleNames[i];
    //        return 2;
    //    }
    //}

    return 0;
}
