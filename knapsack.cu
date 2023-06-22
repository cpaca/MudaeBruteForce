__constant__ char* bestScores;
__device__ char* newBestScores;

// both of these variables are basically the width of each row in bytes
// just host vs device differences
// note that both are only written to ONE TIME (in knapsackInit) and never written to again
__constant__ size_t bestScoresPitch;
size_t host_pitch;
const size_t numRows = MAX_DL + 1;


__host__ void knapsackInit(){
    // named fake row width since host_pitch is the REAL row width
    size_t fakeRowWidth = (OVERLAP_LIMIT + 1) * sizeof(size_t);

    char* host_newBestScores;
    cudaMallocPitch(&host_newBestScores, &host_pitch, fakeRowWidth, numRows);
    cudaMemset(host_newBestScores, 0, host_pitch * numRows);
    cudaMemcpyToSymbol(bestScoresPitch, &host_pitch, sizeof(size_t));
    cudaMemcpyToSymbol(newBestScores, &host_newBestScores, sizeof(char*));

    char* host_bestScores;
    cudaMalloc(&host_bestScores, host_pitch * numRows);
    cudaMemset(host_bestScores, 0, host_pitch * numRows);
    cudaMemcpyToSymbol(bestScores, &host_bestScores, sizeof(char*));
}

__host__ void knapsackReload(){
    cudaMemcpy(bestScores, newBestScores, host_pitch * numRows, cudaMemcpyDeviceToDevice);
}

__device__ size_t knapsackReadBestScore(size_t rowNum, size_t colNum){
    char* rowAddress = bestScores + (rowNum * bestScoresPitch);
    auto* row = (size_t*) rowAddress;
    size_t* col = row + colNum;

    return *col;
}

__device__ bool knapsackIsTaskGood(Task* task){
    size_t DLSlotsRemn = task->DLSlotsRemn;
    size_t DLSlotsUsed = MAX_DL - DLSlotsRemn;

    size_t remainingOverlap = task->remainingOverlap;
    size_t overlapUsed = OVERLAP_LIMIT - remainingOverlap;

    size_t oldBestScore = knapsackReadBestScore(DLSlotsUsed, overlapUsed);

    return true;
}

__device__ void knapsackWrite(const size_t &DLSlotsUsed, const size_t &overlapUsed, const size_t &score){
    size_t rowNum = DLSlotsUsed;
    size_t colNum = overlapUsed;

    char* rowAddress = newBestScores + (rowNum * bestScoresPitch);
    auto* row = (size_t*) rowAddress;
    size_t* col = row + colNum;

    const size_t oldScore = atomicMax(col, score);
    // TODO propagate down the newBestScores knapsack
}

__device__ void knapsackWriteTask(Task* task){
    if(task == nullptr){
        return;
    }

    size_t DLSlotsRemn = task->DLSlotsRemn;
    size_t DLSlotsUsed = MAX_DL - DLSlotsRemn;

    size_t remainingOverlap = task->remainingOverlap;
    size_t overlapUsed = OVERLAP_LIMIT - remainingOverlap;

    knapsackWrite(DLSlotsUsed, overlapUsed, task->score);
}

#undef byte