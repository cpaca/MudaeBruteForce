#ifndef MUDAEBRUTEFORCE_TASK
#define MUDAEBRUTEFORCE_TASK
struct {
    size_t* disabledSets; // List of disabled sets
    // disabledSets[0] to disabledSets[index-1] are defined
    // and disabledSets[index] onwards are undefined
    size_t disabledSetsIndex;
} Task;
#endif