#ifndef MUDAEBRUTEFORCE_TASK
#define MUDAEBRUTEFORCE_TASK
typedef struct {
    // Next index in the setDeleteOrder to attempt deleting
    size_t setDeleteIndex;

    // What the score was the last time it was calculated for this Task
    size_t score;

    // How much OVERLAP_LIMIT is remaining in this Task
    size_t remainingOverlap;

    // How many series/bundles (aka sets) can still be disabled din this Task
    size_t DLSlotsRemn;

    // List of disabled sets
    // disabledSets[0] to disabledSets[index-1] are defined
    // and disabledSets[index] onwards are undefined
    size_t disabledSetsIndex;
    // disabledSets is at the end so that all of the pointers can be together at the end
    size_t* disabledSets;

    // setBundles compatibility
    size_t* bundlesUsed;
} Task;

/**
 * Creates a blank task.
 * The contents of the task are unspecified
 */
__device__ Task* createTask(){
    size_t disabledSetsSize = MAX_DL + MAX_FREE_BUNDLES;

    Task* ret = new Task;
    ret->disabledSets = new size_t[disabledSetsSize];
    ret->bundlesUsed = new size_t[setBundlesSetSize];

    return ret;
}

/**
 * Destructs task. WILL cause a segfault if you use the pointer after destruction.
 */
__device__ void destructTask(Task* task){
    delete[] task->disabledSets;
    delete[] task->bundlesUsed;

    delete task;
}

// Moved to here since it's needed in both Main and TaskQueue
// Moved it here instead of its own file because I couldn't think of a good filename.
/**
 * If set represents the Set ID of a bundle, then bundlesUsed is modified to acknowledge that that bundle is
 * activated.
 * @param numSeries The total number of series there are.
 * @param bundlesUsed MAY BE MODIFIED to acknowledge this set being added to bundlesUsed.
 * @param setToAdd The Set ID of a bundle.
 */
__host__ __device__ void activateBundle(const size_t numSeries, Task* task, size_t set) {
    if(set >= numSeries){
        // setToAdd is actually a bundle to add
        // If this bundle is being used, we need to acknowledge that in bundlesUsed
        size_t bundleNum = set - numSeries;
        size_t bundlesUsedWordSize = 8 * sizeof(size_t);
        size_t bundlesUsedIndex = bundleNum / bundlesUsedWordSize;
        size_t bundleOffset = bundleNum % bundlesUsedWordSize;
        task->bundlesUsed[bundlesUsedIndex] |= (((size_t)1) << bundleOffset);
    }
}
#endif