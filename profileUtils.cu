#define NUM_CLOCKS 4
#define PROFILING_STR_WIDTH 43

// If this is set to false, then no profiling is done
// and you can treat all of the functions in this file like they are empty.
// However, the function signatures are not removed so that you don't have to comment out the functions
// on other files.
#define PROFILE false

// Checkpoints and variables.

__device__ size_t numThreads = 0;

__device__ size_t sharedMemoryCheckpoint = 0;
__device__ size_t syncThreadsCheckpoint = 0;
__device__ size_t whileLoopSetupCheckpoint = 0;
__device__ size_t whileLoopExecutionCheckpoint = 0;
__device__ size_t bundleScoreCheckpoint = 0;
__device__ size_t seriesScoreCheckpoint = 0;
__device__ size_t printValsCheckpoint = 0;

__device__ size_t loopConditionCheckpoint = 0;
__device__ size_t pickSetCheckpoint = 0;
__device__ size_t setSizeCheckpoint = 0;
__device__ size_t bundleOverlapCheckpoint = 0;
__device__ size_t continueLoopCheckpoint = 0;
__device__ size_t addSetToDLCheckpoint = 0;

// The actual computation functions and whatnot.

__device__ size_t* initProfiling(){
#if PROFILE
    atomicAdd(&numThreads, 1);

    auto* clocks = new size_t[NUM_CLOCKS];
    for(size_t i = 0; i < NUM_CLOCKS; i++){
        clocks[i] = -1;
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
    atomicAdd(saveTo, deltaTime);
    // don't reset the clock with clock64()
    // because atomicAdd can take a very, very long time in bad cases
    clocks[clockNum] = clock();
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

__host__ void printProfilingStrNum(const std::string& str, size_t &deviceSymbol, const size_t totalThreads){
#if PROFILE
    size_t num;
    cudaMemcpyFromSymbol(&num, deviceSymbol, sizeof(size_t));
    num /= totalThreads;
    std::cout << padStr(str) << std::to_string(num) << std::endl;
#endif
}

__host__ void printProfilingData(){
#if PROFILE
    size_t totalThreads;
    cudaMemcpyFromSymbol(&totalThreads, numThreads, sizeof(size_t));
    std::cout << "Threads counted: " << std::to_string(totalThreads) << "\n";
    printProfilingStrNum("Avg. time used initializing shared memory: ", sharedMemoryCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used on __syncthreads(): ", syncThreadsCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used setting up the while loop: ", whileLoopSetupCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used executing the while loop: ", whileLoopExecutionCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used calculating bundleScore: ", bundleScoreCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used calculating seriesScore: ", seriesScoreCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used printing vals: ", printValsCheckpoint, totalThreads);
    std::cout << std::endl;
    printProfilingStrNum("Avg. time used checking loop condition: ", loopConditionCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used picking a set: ", pickSetCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used validating set size: ", setSizeCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used validating set bundles: ", bundleOverlapCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used validating set is non-dupe: ", continueLoopCheckpoint, totalThreads);
    printProfilingStrNum("Avg. time used adding set to DL: ", addSetToDLCheckpoint, totalThreads);
    std::cout << std::endl;
#endif
}