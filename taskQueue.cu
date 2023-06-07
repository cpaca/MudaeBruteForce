#include "task.cu"
#define QUEUE_SIZE (1 << 24)

__device__ Task** queue = nullptr;

/**
 * Gets a task from the task queue.
 * If there are no tasks available, returns nullptr.
 * @return
 */
__device__ Task* getTask(){
    // TODO: Implement
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

    convertArrToCuda(host_queue, QUEUE_SIZE * sizeof(size_t*));
    cudaMemcpyToSymbol(queue, &host_queue, sizeof(host_queue));
}