#ifndef MUDAEBRUTEFORCE_TASK
#define MUDAEBRUTEFORCE_TASK
typedef struct {
    size_t* disabledSets; // List of disabled sets
    // disabledSets[0] to disabledSets[index-1] are defined
    // and disabledSets[index] onwards are undefined
    size_t disabledSetsIndex;

    // Next index in the setDeleteOrder to attempt deleting
    size_t setDeleteIndex;
    // Whether or not to delete the next setDeleteOrder index
    bool shouldDeleteNext;

    // What the score was the last time it was calculated for this Task
    size_t score;
} Task;
#endif