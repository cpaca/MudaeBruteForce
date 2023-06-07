#include "task.cu"
#define QUEUE_SIZE (1 << 24)

__device__ Task** queue = nullptr;
__device__ size_t readIdx = 0;
__device__ size_t writeIdx = 0;

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
        queue[queueIdx] = nullptr;
        return ret;
    }
}

/**
 * Puts a task into the task queue.
 * @param task
 */
__device__ void putTask(Task* task){
    // TODO: Implement
}

__host__ void initTaskQueue(){
    // This is a weird way to do it, but doing it this way lets me basically 1:1 repeat other code.
    auto** host_queue = new size_t*[QUEUE_SIZE];
    for(size_t i = 0; i < QUEUE_SIZE; i++){
        // this should be done by default anyway, but this is safer
        host_queue[i] = nullptr;
    }

    convertArrToCuda(host_queue, QUEUE_SIZE);
    cudaMemcpyToSymbol(queue, &host_queue, sizeof(host_queue));
}