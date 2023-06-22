#ifndef MUDAEBRUTEFORCE_KNAPSACK
#define MUDAEBRUTEFORCE_KNAPSACK

__host__ void initSetDeleteOrder(const size_t* host_freeBundles,
                                 size_t** host_bundleData,
                                 size_t** host_seriesData,
                                 size_t numSeries,
                                 size_t numBundles){
    size_t numSets = numSeries + numBundles;
    // since I can assign a "value" to each set, it's easier to use that to compare
    // than define a compare() function
    auto* host_setDeleteOrder = new size_t[numSets];
    auto* host_setDeleteValue = new size_t[numSets];
    for(size_t i = 0; i < numSets; i++){
        size_t setNum = i;
        size_t setValue;
        size_t setSize;
        if(setNum < numSeries){
            setSize = host_seriesData[setNum][0];
        }
        else{
            setSize = host_bundleData[setNum - numSeries][0];
        }
        // Just for now.
        setValue = setSize;

        host_setDeleteOrder[i] = setNum;
        host_setDeleteValue[i] = setValue;
    }
    // From the example:
    // The first array input will be sorted
    // and the second array input will be whatever
    // We want the value to be in sorted order, decreasing order.
    thrust::sort_by_key(host_setDeleteValue,
                        host_setDeleteValue+numSets,
                        host_setDeleteOrder,
                        thrust::greater<size_t>());
    // Convert to CUDA form...
    convertArrToCuda(host_setDeleteOrder, numSets);
    // And take it to CUDA
    cudaMemcpyToSymbol(setDeleteOrder, &host_setDeleteOrder, sizeof(host_setDeleteOrder));
}

// TODO solve the 0-1 knapsack problem

#endif