#ifndef MUDAEBRUTEFORCE_TASK
#define MUDAEBRUTEFORCE_TASK
typedef struct {
    size_t* disabledSets; // List of disabled sets
    // disabledSets[0] to disabledSets[index-1] are defined
    // and disabledSets[index] onwards are undefined
    size_t disabledSetsIndex;

    // Next index in the setDeleteOrder to attempt deleting
    size_t setDeleteIndex;

    // What the score was the last time it was calculated for this Task
    size_t score;

    // How much OVERLAP_LIMIT is remaining in this task
    size_t remainingOverlap;
} Task;

/**
 * Returns a new Task that is identical to the given Task
 * Basically, a copy constructor.
 */
__host__ __device__ Task* copyTask(Task* task){
    if(task == nullptr){
        return nullptr;
    }
    size_t disabledSetsSize = MAX_DL + MAX_FREE_BUNDLES;
    Task* newTask = new Task;
    newTask->disabledSets = new size_t[disabledSetsSize];
    newTask->disabledSetsIndex = task->disabledSetsIndex;
    for(size_t i = 0; i < newTask->disabledSetsIndex; i++){
        newTask->disabledSets[i] = task->disabledSets[i];
    }

    newTask->setDeleteIndex = task->setDeleteIndex;
    newTask->shouldDeleteNext = task->shouldDeleteNext;
    newTask->score = task->score;

    return newTask;
}
#endif