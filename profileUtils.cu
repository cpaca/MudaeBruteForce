#define NUM_CLOCKS 4
#define PROFILING_STR_WIDTH 45

// If this is set to false, then no profiling is done
// and you can treat all of the functions in this file like they are empty.
// However, the function signatures are not removed so that you don't have to comment out the functions
// on other files.
#define PROFILE true

// Checkpoints and variables.

__device__ size_t numThreads = 0;

__device__ size_t getTaskCheckpoint = 0;
__device__ size_t validTaskCheckpoint = 0;
__device__ size_t copyTaskCheckpoint = 0;
__device__ size_t makeNewTaskCheckpoint = 0;
__device__ size_t fullTaskCheckpoint = 0;
__device__ size_t bundlePtrCheckpoint = 0;
__device__ size_t activateBundleCheckpoint = 0;
__device__ size_t deleteSetCheckpoint = 0;
__device__ size_t finishLoopCheckpoint = 0;

__device__ size_t tasksCreated = 0;
__device__ size_t tasksRezzed = 0;

// The actual computation functions and whatnot.

__device__ size_t* initProfiling(){
#if PROFILE
    atomicAdd(&numThreads, 1);

    auto* clocks = new size_t[NUM_CLOCKS];
    for(size_t i = 0; i < NUM_CLOCKS; i++){
        // effectively -1
        // but way easier to detect (or ignore?) problems with
        clocks[i] = ~0;
    }
    return clocks;
#endif
}

__device__ void destructProfiling(const size_t* clocks){
#if PROFILE
    delete[] clocks;
#endif
}

__device__ void startClock(size_t *clocks, int clockNum) {
#if PROFILE
    clocks[clockNum] = clock();
#endif
}

__device__ void checkpoint(size_t *clocks, int clockNum, size_t* saveTo) {
#if PROFILE
    size_t endTime = clock();
    size_t deltaTime = endTime - clocks[clockNum];
    // was getting some errors with profiling giving VERY wrong numbers
    // and it turns out doing this fixes it!
    // Note: 2.5 seconds of computation is less than 10B, so >2Trillion is a really shitty step taking a very long time
    if(deltaTime < 2000000000000L){
        atomicAdd(saveTo, deltaTime);
    }
    // don't reset the clock with clock64()
    // because atomicAdd can take a very, very long time in bad cases
    clocks[clockNum] = clock();
#endif
}

/**
 * Increments a certain value, if the profiler is running.
 * This is helpful for saying how many times certain things happened
 * but this is also a profiler thing so it self-disables when you stop profiling
 */
__device__ void profileIncrement(size_t* saveTo){
#if PROFILE
    atomicAdd(saveTo, 1);
#endif
}

__host__ std::string padStr(const std::string& str){
#if PROFILE
    int missingLength = PROFILING_STR_WIDTH - str.length();
    if(missingLength <= 0){
        return str;
    }
    return str + std::string(missingLength, ' ');
#else
    return "";
#endif
}

__host__ void printProfilingStrNum(const std::string& str, size_t &deviceSymbol, const size_t totalThreads = 0){
#if PROFILE
    size_t num;
    cudaMemcpyFromSymbol(&num, deviceSymbol, sizeof(size_t));
    if(totalThreads != 0) {
        num /= totalThreads;
    }
    std::cout << padStr(str) << std::to_string(num) << std::endl;
#endif
}

__host__ void printProfilingData(){
#if PROFILE
    size_t totalThreads;
    cudaMemcpyFromSymbol(&totalThreads, numThreads, sizeof(size_t));
    std::cout << "Threads counted: " << std::to_string(totalThreads) << "\n";
    printProfilingStrNum("Avg. time used getting the task: ", getTaskCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used validating the task: ", validTaskCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used copying new task: ", copyTaskCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used initializing new task: ", makeNewTaskCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used deleting an overfilled set: ", fullTaskCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used getting the bundle ptr: ", bundlePtrCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used deleting the bundle's series: ", activateBundleCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used finishing deleteSet: ", deleteSetCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used postprocessing: ", finishLoopCheckpoint, totalThreads);
    std::cout << "\n";
    printProfilingStrNum("Number of tasks created: ", tasksCreated);
    printProfilingStrNum("Number of tasks resurrected: ", tasksRezzed);
    std::cout << "\n";

    TaskQueue host_deadTaskQueue;
    cudaMemcpyFromSymbol(&host_deadTaskQueue, deadTaskQueue, sizeof(TaskQueue));
    std::cout << "Dead queue fullness: " << std::to_string(host_deadTaskQueue.writeIdx - host_deadTaskQueue.readIdx) << "\n";
    std::cout << "Dead queue writeIdx: " << std::to_string(host_deadTaskQueue.writeIdx) << "\n";


    std::cout << std::endl;
#endif
}