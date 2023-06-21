#include "task.cu"
#include <thrust/sort.h>
#define LIVE_QUEUE_SIZE 24

typedef struct {
    Task** queue;

    // The length of the queue is 2 to the power of size
    // so if size is 24, then the length of the queue is (1 << 24)
    std::uint8_t size;
    size_t readIdx;
    size_t writeIdx;
} TaskQueue;

// Queue for live tasks
__device__ TaskQueue liveTaskQueue;

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
    size_t disabledSetsSize = MAX_DL + MAX_FREE_BUNDLES;
    // TODO use deadTaskQueue
    Task* newTask = createTask();

    memcpy(newTask, task, sizeof(Task) - (2 * sizeof(size_t*)));
    memcpy(newTask->disabledSets, task->disabledSets, sizeof(size_t) * disabledSetsSize);
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
    // TODO use deadTaskQueue
    destructTask(task);
}

__host__ void initTaskQueue(const size_t* host_freeBundles,
                            size_t** host_bundleData,
                            size_t** host_seriesData,
                            size_t numSeries,
                            size_t numBundles){
    size_t disabledSetsSize = MAX_DL + MAX_FREE_BUNDLES;
    size_t numSets = numSeries + numBundles;

    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    TaskQueue host_liveTaskQueue;
    host_liveTaskQueue.queue = new Task*[LIVE_QUEUE_SIZE];
    for(size_t i = 0; i < LIVE_QUEUE_SIZE; i++){
        // this should be done by default anyway, but this is safer
        host_liveTaskQueue.queue[i] = nullptr;
    }

    // Also create a very basic task for the very first thread.
    Task* firstTask = new Task;
    firstTask->disabledSets = new size_t[disabledSetsSize];
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
    convertArrToCuda(firstTask->disabledSets, disabledSetsSize);
    convertArrToCuda(firstTask, 1);
    host_liveTaskQueue.queue[0] = firstTask;

    convertArrToCuda(host_liveTaskQueue.queue, (1 << LIVE_QUEUE_SIZE));
    host_liveTaskQueue.size = LIVE_QUEUE_SIZE;
    host_liveTaskQueue.readIdx = 0;
    host_liveTaskQueue.writeIdx = 1;
    cudaMemcpyToSymbol(liveTaskQueue, &host_liveTaskQueue, sizeof(TaskQueue));
    // Clean up memory
    // Looks like this is the only one which doesn't get sent to CUDA.
    delete[] freeBundlePtrs;
}