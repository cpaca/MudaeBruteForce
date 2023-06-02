#define NUM_CLOCKS 4
#define PROFILING_STR_WIDTH 43

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
    atomicAdd(&numThreads, 1);

    auto* clocks = new size_t[NUM_CLOCKS];
    for(size_t i = 0; i < NUM_CLOCKS; i++){
        clocks[i] = -1;
    }
    return clocks;
}

__device__ void destructProfiling(const size_t* clocks){
    delete[] clocks;
}

__device__ void startClock(size_t *clocks, int clockNum) {
    clocks[clockNum] = clock();
}

__device__ void checkpoint(size_t *clocks, int clockNum, size_t* saveTo) {
    size_t endTime = clock();
    size_t deltaTime = endTime - clocks[clockNum];
    atomicAdd(saveTo, deltaTime);
    // don't reset the clock with clock64()
    // because atomicAdd can take a very, very long time in bad cases
    clocks[clockNum] = clock();
}

__host__ std::string padStr(const std::string& str){
    int missingLength = PROFILING_STR_WIDTH - str.length();
    if(missingLength <= 0){
        return str;
    }
    return str + std::string(missingLength, ' ');
}

__host__ void printProfilingStrNum(const std::string& str, size_t &deviceSymbol, const size_t totalThreads){
    size_t num;
    cudaMemcpyFromSymbol(&num, deviceSymbol, sizeof(size_t));
    num /= totalThreads;
    std::cout << padStr(str) << std::to_string(num) << std::endl;
}

__host__ void printProfilingData(){
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
}