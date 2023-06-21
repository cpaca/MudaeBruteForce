#ifndef MUDAEBRUTEFORCE_TYPES
#define MUDAEBRUTEFORCE_TYPES
typedef std::uint16_t setSize_t;

template <typename T>
/**
 * Gets the maximum value for an unsigned value. Uses the method described in https://stackoverflow.com/a/39878362
 */
__host__ __device__ T get_unsigned_max(){
    return ~(static_cast<T>(0));
}

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

typedef struct {
    Task** queue;

    // The length of the queue is 2 to the power of size
    // so if size is 24, then the length of the queue is (1 << 24)
    std::uint8_t size;
    size_t readIdx;
    size_t writeIdx;
} TaskQueue;
#endif