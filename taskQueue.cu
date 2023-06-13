#include "task.cu"
#include <thrust/sort.h>
#define QUEUE_SIZE (1 << 24)

__device__ Task** queue = nullptr;
__device__ size_t readIdx = 0;
__device__ size_t writeIdx = 1; // we start with exactly 1 task

/**
 * Gets a task from the task queue.
 * If there are no tasks available, returns nullptr.
 * @return
 */
__device__ Task* getTask(){
    while(true){
        size_t expectedReadIdx = readIdx;
        if(readIdx >= writeIdx){
            return nullptr;
        }
        // Otherwise, attempt to get the read idx...
        size_t atomicReadIdx = atomicCAS(&readIdx, expectedReadIdx, expectedReadIdx+1);
        if(atomicReadIdx != expectedReadIdx){
            // Some other thread got the expectedReadIdx task, so we can't.
            continue;
        }
        // This thread got the expectedReadIdx.
        size_t queueIdx = expectedReadIdx % QUEUE_SIZE;
        Task* ret = queue[queueIdx];
        while(ret == nullptr){
            // Apparantly the writeIdx got incremented but putTask wasn't ready.
            ret = queue[queueIdx];
        }
        queue[queueIdx] = nullptr;
        return ret;
    }
}

/**
 * Puts a task into the task queue.
 * WARNING: USING OR TOUCHING THE TASK AFTER CALLING PUTTASK() IS **UNDEFINED BEHAVIOR**
 * (Because getTask could do something with it in another thread)
 */
__device__ void putTask(Task* task){
    size_t putIdx = atomicAdd(&writeIdx, 1);
    size_t queueIdx = putIdx % QUEUE_SIZE;
    queue[queueIdx] = task;
}

__host__ void initTaskQueue(const size_t* host_freeBundles,
                            size_t** host_bundleData,
                            size_t** host_seriesData,
                            size_t numSeries,
                            size_t numBundles){
    size_t disabledSetsSize = MAX_DL + MAX_FREE_BUNDLES;
    size_t numSets = numSeries + numBundles;

    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    auto** host_queue = new Task*[QUEUE_SIZE];
    for(size_t i = 0; i < QUEUE_SIZE; i++){
        // this should be done by default anyway, but this is safer
        host_queue[i] = nullptr;
    }

    // Also create a very basic task for the very first thread.
    Task* firstTask = new Task;
    firstTask->disabledSets = new size_t[disabledSetsSize];
    firstTask->disabledSetsIndex = 0;
    for(size_t i = 0; i < numBundles; i++){
        if(host_freeBundles[i] != 0){
            // Add this free bundle to disabledSets.
            firstTask->disabledSets[firstTask->disabledSetsIndex] = numSeries + i;
            firstTask->disabledSetsIndex++;
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

    convertArrToCuda(firstTask->disabledSets, disabledSetsSize);
    // There is a very small chance that this works.
    convertArrToCuda(firstTask, 1);
    host_queue[0] = firstTask;

    convertArrToCuda(host_queue, QUEUE_SIZE);
    cudaMemcpyToSymbol(queue, &host_queue, sizeof(host_queue));
}