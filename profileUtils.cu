#define NUM_CLOCKS 4

// Checkpoints.

__device__ size_t sharedMemoryCheckpoint;

// The actual computation functions and whatnot.

__device__ size_t* clocks = nullptr;

__device__ void initProfiling(){
    clocks = new size_t[NUM_CLOCKS];
    for(size_t i = 0; i < NUM_CLOCKS; i++){
        clocks[i] = -1;
    }
}

__device__ void destructProfiling(){
    delete[] clocks;
}

__device__ void startClock(int clockNum){
    clocks[clockNum] = clock64();
}

__device__ void checkpoint(int clockNum, size_t& saveTo){
    size_t endTime = clock64();
    size_t deltaTime = endTime - clocks[clockNum];
    atomicAdd(&saveTo, deltaTime);
    // don't reset the clock with clock64()
    // because atomicAdd can take a very, very long time in bad cases
    clocks[clockNum] = clock64();
}

__host__ void printProfilingStrNum(const std::string& str, size_t &deviceSymbol){
    size_t num;
    cudaMemcpyFromSymbol(&num, deviceSymbol, sizeof(size_t));
    std::cout << str << std::to_string(num) << std::endl;
}

__host__ void printProfilingData(){
    printProfilingStrNum("Time used initializing shared memory: ", sharedMemoryCheckpoint);
}