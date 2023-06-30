#include "task.cu"
#include <thrust/sort.h>
#define QUEUE_SIZE 24
#define QUEUE_ELEMENTS (((size_t) 1) << QUEUE_SIZE)

/**
 * Gets a task from the task queue.
 * If there are no tasks available, returns nullptr.
 * @return
 */
__device__ Task* getTask(TaskQueue &tasks){
    // TODO rewrite using atomicAdd
    //  - low priority, is just an optimization idea
    size_t offset = (threadIdx.x % 32) + 1;
    while(true){
        offset = min(offset, offset-1);
        size_t expectedReadIdx = tasks.readIdx + offset;
        if(expectedReadIdx > tasks.writeIdx){
            return nullptr;
        }

        // Attempt to acquire this read idx...
        size_t atomicReadIdx = atomicCAS(&(tasks.readIdx), expectedReadIdx, expectedReadIdx+1);
        if(atomicReadIdx != expectedReadIdx){
            // Some other thread got the readIdx, so we gotta try again.
            continue;
        }

        char* queueAddress = (char*) tasks.queue;
        size_t queueIdx = expectedReadIdx % QUEUE_ELEMENTS;
        char* taskAddress = queueAddress + (queueIdx * queuePitch);
        Task* ret = (Task*) taskAddress;
        return ret;
    }
}

/**
 * Puts a task into the task queue.
 * WARNING: USING OR TOUCHING THE TASK AFTER CALLING PUTTASK() IS **UNDEFINED BEHAVIOR**
 * (Because getTask could do something with it in another thread)
 */
__device__ void putTask(TaskQueue &tasks, Task* task){
    // TODO reimplement
}

__host__ TaskQueue makeBlankTaskQueue(size_t queueSize){
    size_t taskStructBytes = sizeof(Task);
    size_t bundlesUsedBytes = sizeof(size_t) * host_setBundlesSetSize;
    size_t disabledSetsBytes = DISABLED_SETS_SIZE * sizeof(size_t);
    size_t taskTotalBytes = taskStructBytes+bundlesUsedBytes+disabledSetsBytes;

    TaskQueue ret;

    size_t host_queuePitch;
    cudaMallocPitch(&ret.queue, &host_queuePitch, taskTotalBytes, QUEUE_ELEMENTS);
    cudaMemcpyToSymbol(queuePitch, &host_queuePitch, sizeof(host_queuePitch));

    ret.readIdx = 0;
    ret.writeIdx = 0;
    return ret;
}

/**
 * Copies the outTaskQueue to the inTaskQueue
 */
__host__ void reloadTaskQueue(){
    knapsackReload();

    // Get and swap the two task queues
    // Note that device inTaskQueue goes to host outTaskQueue
    TaskQueue host_inTaskQueue;
    TaskQueue host_outTaskQueue;
    cudaMemcpyFromSymbol(&host_outTaskQueue, inTaskQueue, sizeof(TaskQueue));
    cudaMemcpyFromSymbol(&host_inTaskQueue, outTaskQueue, sizeof(TaskQueue));

    // Save data for debug...
    size_t numTasks = host_inTaskQueue.writeIdx - host_inTaskQueue.readIdx;

    // Reset...
    host_outTaskQueue.readIdx = 0;
    host_outTaskQueue.writeIdx = 0;

    // Reload...
    cudaMemcpyToSymbol(inTaskQueue, &host_inTaskQueue, sizeof(TaskQueue));
    cudaMemcpyToSymbol(outTaskQueue, &host_outTaskQueue, sizeof(TaskQueue));

    // Update the expected setDeleteIndex
    setDeleteIndex++;
    size_t host_expectedSetToDelete = host_setDeleteOrder[setDeleteIndex];
    cudaMemcpyToSymbol(expectedSetToDelete, &host_expectedSetToDelete, sizeof(size_t));

    // Print some stuff for debug reasons
    std::cout << "With a setDeleteIndex of " << std::to_string(setDeleteIndex) << ",\n";
    std::cout << "the inTaskQueue has " << std::to_string(numTasks) << " tasks\n" << std::endl;
}

__host__ void initTaskQueue(const size_t* host_freeBundles,
                            size_t** host_bundleData,
                            size_t** host_seriesData,
                            size_t numSeries,
                            size_t numBundles){
    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    TaskQueue host_inTaskQueue = makeBlankTaskQueue(QUEUE_SIZE);

    // Also create a very basic task for the very first thread.
    // TODO maybe make this a one-thread kernel call?
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
    cudaMemcpy(host_inTaskQueue.queue, firstTask, sizeof(Task), cudaMemcpyHostToDevice);

    convertArrToCuda(host_inTaskQueue.queue, QUEUE_ELEMENTS);
    host_inTaskQueue.readIdx = 0;
    host_inTaskQueue.writeIdx = 1;
    cudaMemcpyToSymbol(inTaskQueue, &host_inTaskQueue, sizeof(TaskQueue));

    TaskQueue host_deadTaskQueue = makeBlankTaskQueue(QUEUE_SIZE);
    cudaMemcpyToSymbol(deadTaskQueue, &host_deadTaskQueue, sizeof(TaskQueue));

    TaskQueue host_outTaskQueue = makeBlankTaskQueue(QUEUE_SIZE);
    cudaMemcpyToSymbol(outTaskQueue, &host_outTaskQueue, sizeof(TaskQueue));

    // Clean up memory
    // Looks like this is the only one which doesn't get sent to CUDA.
    delete[] freeBundlePtrs;
    // TODO check if delete firstTask is safe
}