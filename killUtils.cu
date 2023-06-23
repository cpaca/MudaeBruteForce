#ifndef MUDAEBRUTEFORCE_KILLUTILS
#define MUDAEBRUTEFORCE_KILLUTILS

/**
 * Measures if the score gained for a certain amount of overlap used is efficient.
 * If it's not efficient, you should probably kill the task.
 */
__device__ bool isEfficient(const size_t &scoreGained, const size_t &overlapUsed){
    // Current decision:
    // A bundle needs an efficiency of at least 20% to be efficient.
    // That means (score gained) > (20% * overlapUsed)
    // Or (score gained) > (overlapUsed / 5)
    size_t scoreReq = overlapUsed/5;
    return scoreGained > scoreReq;
}

/**
 * Tells you if you should kill the Task.
 * Note that all parameters should be non-nullptrs.
 * @param prevTask The version of the task before the latest set got added.
 * @param task The version of the task after the latest set got added.
 * @return If you should kill the task.
 */
__device__ bool shouldKill(Task* prevTask, Task* task){
    if(task->score == prevTask->score) {
        // Performing this Task didn't add any value...
        // So we shouldn't continue it
        return true;
    }
    if (!isEfficient(task->score - prevTask->score, prevTask->remainingOverlap - task->remainingOverlap)){
        // Is NOT efficient
        // Kill!
        return true;
    }
    return false;
}

#endif