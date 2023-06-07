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
    // TODO: Implement
}