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
 */
__device__ void putTask(TaskQueue &tasks, Task* task){
    if(task == nullptr){
        return;
    }
    size_t putIdx = atomicAdd(&(tasks.writeIdx), 1);
    size_t queueIdx = putIdx % QUEUE_ELEMENTS;
    char* queueAddress = (char*) tasks.queue;
    char* taskAddress = queueAddress + (queueIdx * queuePitch);

    char* destAddress = taskAddress;
    char* srcAddress = (char*) task;
    memcpy(destAddress, srcAddress, queuePitch);
}

__host__ TaskQueue makeBlankTaskQueue() {
    // This is VERY VERY BAD programming practice to have repeated code in makeBlankTaskQueue and in kernelInitTaskQueue
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

// The kernel-side function that assists with initializing the taskQueue
__global__ void kernelInitTaskQueue(size_t numSeries, size_t numBundles){
    // This is VERY VERY BAD programming practice to have repeated code in makeBlankTaskQueue and in kernelInitTaskQueue
    size_t taskStructBytes = sizeof(Task);
    size_t bundlesUsedBytes = sizeof(size_t) * setBundlesSetSize;
    size_t disabledSetsBytes = DISABLED_SETS_SIZE * sizeof(size_t);
    size_t taskTotalBytes = taskStructBytes+bundlesUsedBytes+disabledSetsBytes;

    // Create and init task:
    // Init malloc-related stuff
    char* baseAddress = (char*) malloc(queuePitch);
    devicePrintStrNum("baseAddress: ", (size_t) baseAddress);
    Task* taskAddress = (Task*) (baseAddress);
    auto* bundlesUsedAddress = (size_t*) (baseAddress + taskStructBytes);
    auto* disabledSetsAddress = (size_t*) (baseAddress + taskStructBytes + disabledSetsBytes);

    taskAddress->bundlesUsed = bundlesUsedAddress;
    taskAddress->disabledSets = disabledSetsAddress;

    // Init simple constants
    taskAddress->remainingOverlap = OVERLAP_LIMIT;
    taskAddress->DLSlotsRemn = MAX_DL;
    taskAddress->disabledSetsIndex = 0;
    taskAddress->score = 0;

    // Init complex constants
    // Proper init score and disabledSetsIndex... and disabledSets
    auto** bundlePtrs = new size_t*[numBundles];
    for(size_t i = 0; i < numBundles; i++){
        if(freeBundles[i] != 0){
            size_t setNum = numSeries + i;
            activateBundle(numSeries, taskAddress, setNum);
            taskAddress->disabledSets[taskAddress->disabledSetsIndex] = setNum;
            bundlePtrs[taskAddress->disabledSetsIndex] = bundleSeries + bundleIndices[i];
            taskAddress->disabledSetsIndex++;
        }
    }

    for(size_t seriesNum = 0; seriesNum < numSeries; seriesNum++){
        bool addSeries = false;
        for(size_t i = 0; i < taskAddress->disabledSetsIndex; i++){
            size_t bundleSetNum = *(bundlePtrs[i]);
            while(bundleSetNum < seriesNum){
                bundlePtrs[i]++;
                bundleSetNum = *(bundlePtrs[i]);
            }
            if(bundleSetNum == seriesNum){
                addSeries = true;
            }
        }

        if(addSeries){
            taskAddress->score += deviceSeries[(2 * seriesNum) + 1];
        }
    }
    delete[] bundlePtrs;

    putTask(outTaskQueue, taskAddress);

    auto* queue = (std::uint8_t*) outTaskQueue.queue;
    printf("kernelInitTaskQueue, printing bytes:\n");
    for(size_t byteNum = 0; byteNum < taskTotalBytes; byteNum++){
        devicePrintStrNum("", queue[byteNum], 16, 0, true);
    }

    free(baseAddress);
}

__host__ void initTaskQueue(size_t numSeries, size_t numBundles){
    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    TaskQueue host_inTaskQueue = makeBlankTaskQueue();
    cudaMemcpyToSymbol(inTaskQueue, &host_inTaskQueue, sizeof(TaskQueue));

    TaskQueue host_outTaskQueue = makeBlankTaskQueue();
    cudaMemcpyToSymbol(outTaskQueue, &host_outTaskQueue, sizeof(TaskQueue));

    kernelInitTaskQueue<<<1, 1>>>(numSeries, numBundles);
    cudaError_t syncError = cudaDeviceSynchronize();
    if(syncError != cudaSuccess){
        std::cout << "initTaskQueue invoked a CUDA error" << std::endl;
        const char *errName = cudaGetErrorName(syncError);
        printf("%s\n", errName);
        const char *errStr = cudaGetErrorString(syncError);
        printf("%s\n", errStr);
        std::cout << std::endl;
        assert(false);
    }

    // Since kernelInitTaskQueue calls putTask and I plan to make putTask go to outQueue only
    // this is how I have to swap the input and output sides.
    std::cout << "InitTaskQueue calling Reload";
    reloadTaskQueue();
}