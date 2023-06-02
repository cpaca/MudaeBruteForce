#define NUM_CLOCKS 4

// Checkpoints and variables.

__device__ size_t numThreads = 0;

__device__ size_t sharedMemoryCheckpoint = 0;

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

__host__ void printProfilingStrNum(const std::string& str, size_t &deviceSymbol, const size_t totalThreads){
    size_t num;
    cudaMemcpyFromSymbol(&num, deviceSymbol, sizeof(size_t));
    num /= totalThreads;
    std::cout << str << std::to_string(num) << std::endl;
}

__host__ void printProfilingData(){
    size_t totalThreads;
    cudaMemcpyFromSymbol(&totalThreads, numThreads, sizeof(size_t));
    std::cout << "Threads counted: " << std::to_string(totalThreads) << "\n";
    printProfilingStrNum("Avg. time used initializing shared memory: ", sharedMemoryCheckpoint, totalThreads);
}