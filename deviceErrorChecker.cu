#include <cassert>

__host__ void cudaErrorCheck(cudaError_t error, const std::string& str) {
    if (error != cudaSuccess) {
        std::cout << "Caught a CUDA error. Message: " << str << "\n";
        std::cout << "Error type: " << cudaGetErrorName(error) << "\n";
        std::cout << "Error string: " << cudaGetErrorString(error) << std::endl;

        assert(false);
    }
}