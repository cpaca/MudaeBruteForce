#include "task.cu"
#include <thrust/sort.h>
#define LIVE_QUEUE_SIZE 24
#define DEAD_QUEUE_SIZE 15

/**
 * Gets a task from the task queue.
 * If there are no tasks available, returns nullptr.
 * @return
 */
__device__ Task* getTask(TaskQueue &tasks){
    size_t offset = (threadIdx.x % 32) + 1;
    while(true){
        offset = min(offset, offset-1);

        size_t expectedReadIdx = tasks.readIdx + offset;
        if(expectedReadIdx >= tasks.writeIdx){
            return nullptr;
        }

        size_t queueIdx = expectedReadIdx % (1 << tasks.size);
        Task* ret = tasks.queue[queueIdx];
        if(ret == nullptr){
            // putTask is in the process of putting the task in.
            // So pick it up next time.
            return nullptr;
        }

        // Otherwise, attempt to get the read idx...
        size_t atomicReadIdx = atomicCAS(&(tasks.readIdx), expectedReadIdx, expectedReadIdx+1);
        if(atomicReadIdx != expectedReadIdx){
            // Some other thread got the expectedReadIdx task, so we can't.
            continue;
        }
        tasks.queue[queueIdx] = nullptr;
        return ret;
    }
}

/**
 * Puts a task into the task queue.
 * WARNING: USING OR TOUCHING THE TASK AFTER CALLING PUTTASK() IS **UNDEFINED BEHAVIOR**
 * (Because getTask could do something with it in another thread)
 */
__device__ void putTask(TaskQueue &tasks, Task* task){
    if(task == nullptr){
        return;
    }
    size_t putIdx = atomicAdd(&(tasks.writeIdx), 1);
    size_t queueIdx = putIdx % (1 << tasks.size);
    tasks.queue[queueIdx] = task;
}

/**
 * Returns a new Task that is identical to the given Task
 * Basically, a copy constructor.
 */
__device__ Task* copyTask(Task* task){
    if(task == nullptr){
        return nullptr;
    }
    Task* newTask = getTask(deadTaskQueue);
    if(newTask == nullptr) {
        newTask = createTask();
        profileIncrement(&tasksCreated);
    }
    else{
        profileIncrement(&tasksRezzed);
    }

    memcpy(newTask, task, sizeof(Task) - (2 * sizeof(size_t*)));
    // Absolutely GENIUS optimization
    // We don't care about anything from disabledSetsIndex onward so we don't have to copy it
    memcpy(newTask->disabledSets, task->disabledSets, sizeof(size_t) * task->disabledSetsIndex);
    memcpy(newTask->bundlesUsed, task->bundlesUsed, sizeof(size_t) * setBundlesSetSize);

    return newTask;
}

/**
 * Deletes a task.
 * Treat this like you would a destructor; if you call this then DON'T TOUCH THE TASK FOR ANY REASON
 * Unless you want a segfault.
 * Renamed to killTask because when it's killed, it goes into the dead task queue ("dead". "kill".)
 */
__device__ void killTask(Task* task){
    size_t queueFullness = deadTaskQueue.writeIdx - deadTaskQueue.readIdx;
    if(queueFullness >= (1 << DEAD_QUEUE_SIZE)){
        // Too many dead tasks, just really truly kill this one
        destructTask(task);
    }
    else{
        // Kill this one for resurrection later.
        putTask(deadTaskQueue, task);
    }
}

__host__ TaskQueue makeBlankTaskQueue(size_t queueSize){
    TaskQueue ret;
    ret.queue = new Task*[1 << queueSize];
    for(size_t i = 0; i < (1 << queueSize); i++){
        // this should be done by default anyway, but this is safer
        ret.queue[i] = nullptr;
    }
    convertArrToCuda(ret.queue, (1 << queueSize));
    ret.size = queueSize;
    ret.readIdx = 0;
    ret.writeIdx = 0;
    return ret;
}

__host__ void initTaskQueue(const size_t* host_freeBundles,
                            size_t** host_bundleData,
                            size_t** host_seriesData,
                            size_t numSeries,
                            size_t numBundles){
    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    TaskQueue host_liveTaskQueue;
    host_liveTaskQueue.queue = new Task*[1 << LIVE_QUEUE_SIZE];
    for(size_t i = 0; i < (1 << LIVE_QUEUE_SIZE); i++){
        // this should be done by default anyway, but this is safer
        host_liveTaskQueue.queue[i] = nullptr;
    }

    // Also create a very basic task for the very first thread.
    Task* firstTask = new Task;
    firstTask->disabledSets = new size_t[DISABLED_SETS_SIZE];
    firstTask->disabledSetsIndex = 0;
    // Will need the free bundle pointers for score calculations.
    auto** freeBundlePtrs = new size_t*[numBundles];
    size_t numFreeBundles = 0;
    for(size_t i = 0; i < numBundles; i++){
        if(host_freeBundles[i] != 0){
            // Add this free bundle to disabledSets.
            firstTask->disabledSets[firstTask->disabledSetsIndex] = numSeries + i;
            firstTask->disabledSetsIndex++;

            // Note free bundle in pointers:
            // And skip the size value.
            freeBundlePtrs[numFreeBundles] = (host_bundleData[i])+1;
            numFreeBundles++;
        }
    }

    // Initialize setDeleteInformation
    firstTask->setDeleteIndex = 0;

    // Initialize score
    size_t score = 0;
    for(size_t i = 0; i < numSeries; i++){
        bool addSeries = false;
        for(size_t j = 0; j < numFreeBundles; j++){
            if((*(freeBundlePtrs[j])) == i){
                freeBundlePtrs[j]++;
                addSeries = true;
            }
        }
        if(addSeries){
            score += host_seriesData[i][1];
        }
    }
    firstTask->score = score;

    // Initialize setBundles
    firstTask->bundlesUsed = new size_t[host_setBundlesSetSize];
    for(size_t i = 0; i < host_setBundlesSetSize; i++){
        // otherwise it's filled with random 1s and 0s and there are problems
        firstTask->bundlesUsed[i] = 0;
    }
    for(size_t i = 0; i < firstTask->disabledSetsIndex; i++){
        activateBundle(numSeries, firstTask, firstTask->disabledSets[i]);
    }
    convertArrToCuda(firstTask->bundlesUsed, host_setBundlesSetSize);

    // Initialize constant variables
    firstTask->remainingOverlap = OVERLAP_LIMIT;
    firstTask->DLSlotsRemn = MAX_DL;

    // Convert everything into CUDA form:
    convertArrToCuda(firstTask->disabledSets, DISABLED_SETS_SIZE);
    convertArrToCuda(firstTask, 1);
    host_liveTaskQueue.queue[0] = firstTask;

    convertArrToCuda(host_liveTaskQueue.queue, (1 << LIVE_QUEUE_SIZE));
    host_liveTaskQueue.size = LIVE_QUEUE_SIZE;
    host_liveTaskQueue.readIdx = 0;
    host_liveTaskQueue.writeIdx = 1;
    cudaMemcpyToSymbol(liveTaskQueue, &host_liveTaskQueue, sizeof(TaskQueue));

    TaskQueue host_deadTaskQueue = makeBlankTaskQueue(DEAD_QUEUE_SIZE);
    cudaMemcpyToSymbol(deadTaskQueue, &host_deadTaskQueue, sizeof(TaskQueue));

    // Clean up memory
    // Looks like this is the only one which doesn't get sent to CUDA.
    delete[] freeBundlePtrs;
}