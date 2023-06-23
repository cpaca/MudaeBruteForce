#ifndef MUDAEBRUTEFORCE_KNAPSACK
#define MUDAEBRUTEFORCE_KNAPSACK

__host__ size_t** getFreeBundlePtrs(const size_t* host_freeBundles,
                                size_t** host_bundleData,
                                size_t numBundles,
                                size_t &numFreeBundles,
                                bool passSizes = false){
    auto** ret = new size_t*[numBundles];
    numFreeBundles = 0;

    for(size_t i = 0; i < numBundles; i++){
        if(host_freeBundles[i] != 0){
            // This one is a free bundle.
            ret[numFreeBundles] = host_bundleData[i];
            numFreeBundles++;
        }
    }
    ret[numFreeBundles] = nullptr;

    if(passSizes){
        // get past the sizes
        for(size_t i = 0; i < numFreeBundles; i++){
            ret[i]++;
        }
    }

    return ret;
}

__host__ void initSetDeleteOrder(const size_t* host_freeBundles,
                                 size_t** host_bundleData,
                                 size_t** host_seriesData,
                                 size_t numSeries,
                                 size_t numBundles){
    size_t numSets = numSeries + numBundles;
    // since I can assign a "value" to each set, it's easier to use that to compare
    // than define a compare() function
    host_setDeleteOrder = new size_t[numSets];
    auto* host_setDeleteValue = new size_t[numSets];
    // Score gained from deleting a set.
    // Accounts for freeBundles, but is otherwise naive.
    auto* host_setDeleteScore = new size_t[numSets];

    size_t numScoresToNote = 10;
    auto* scoresToNote = (size_t*) malloc(numScoresToNote * sizeof(size_t));
    memset(scoresToNote, 0, numScoresToNote * sizeof(size_t));
    size_t unnotedScores = 0;

    for(size_t setNum = 0; setNum < numSets; setNum++){
        size_t setValue;
        size_t setSize;
        if(setNum < numSeries){
            setSize = host_seriesData[setNum][0];
        }
        else{
            setSize = host_bundleData[setNum - numSeries][0];
        }

        // Calculate the score gained from deleting this set.
        size_t numFreeBundles;
        size_t** freeBundlePtrs = getFreeBundlePtrs(host_freeBundles, host_bundleData, numBundles, numFreeBundles, true);
        size_t setDeleteScore;
        if(setNum < numSeries){
            setDeleteScore = host_seriesData[setNum][1];
            for(size_t freeBundleNum = 0; freeBundleNum < numFreeBundles; freeBundleNum++){
                size_t* freeBundlePtr = freeBundlePtrs[freeBundleNum];
                while(*freeBundlePtr < setNum){
                    // Before the set in question, just ignore
                    // note that since it's unsigned type, the -1 value will also stop this loop
                    freeBundlePtr++;
                }
                if((*freeBundlePtr) == setNum){
                    setDeleteScore = 0;
                    // we know the score is 0 so this for loop is done
                    break;
                }
            }
        }
        else{
            // Is a bundle.
            size_t bundleNum = setNum - numSeries;
            size_t* bundlePtr = host_bundleData[bundleNum];
            bundlePtr++;
            setDeleteScore = 0;
            while((*bundlePtr) != -1){
                size_t seriesNum = *bundlePtr;
                bool addSeriesScore = true;

                for(size_t freeBundleNum = 0; freeBundleNum < numFreeBundles; freeBundleNum++){
                    while((*freeBundlePtrs[freeBundleNum]) < seriesNum){
                        freeBundlePtrs[freeBundleNum]++;
                    }
                    if((*freeBundlePtrs[freeBundleNum]) == seriesNum){
                        addSeriesScore = false;
                    }
                }

                if(addSeriesScore){
                    setDeleteScore += host_seriesData[seriesNum][1];
                }

                // And next series.
                bundlePtr++;
            }
        }

        // Delete sets based on their rewarded score.
        setValue = setDeleteScore;
        setValue *= 32768;
        // With a size of 1 this adds a value of:
        //  32768 - min(1, 32767)
        //  32768 - 1
        //  32767
        // With a size of infinity this adds a value of:
        //  32768 - min(inf, 32767)
        //  32768 - 32767
        //  1
        // So this makes smaller sets SLIGHTLY better than bigger ones.
        setValue += 32768 - std::min(setSize, (size_t) 32767);

        host_setDeleteOrder[setNum] = setNum;
        host_setDeleteValue[setNum] = setValue;
        host_setDeleteScore[setNum] = setDeleteScore;

        // Note notable stuff
        if(setDeleteScore < numScoresToNote){
            scoresToNote[setDeleteScore]++;
        }
        else{
            unnotedScores++;
        }
    }
    // From the example:
    // The first array input will be sorted
    // and the second array input will be whatever
    // We want the value to be in sorted order, decreasing order.
    thrust::sort_by_key(host_setDeleteValue,
                        host_setDeleteValue+numSets,
                        host_setDeleteOrder,
                        thrust::greater<size_t>());
    // And take it to CUDA
    cudaMemcpyToSymbol(expectedSetToDelete, &host_setDeleteOrder[0], sizeof(size_t));

    // Also state anything notable
    for(size_t i = 0; i < numScoresToNote; i++){
        std::cout << "There were " << std::to_string(scoresToNote[i]) << " sets with a size of " << std::to_string(i) << "\n";
    }
    std::cout << "There were " << std::to_string(unnotedScores) << " other sets\n";
    std::cout << std::endl;
    delete[] scoresToNote;
}

// TODO solve the 0-1 knapsack problem

#endif