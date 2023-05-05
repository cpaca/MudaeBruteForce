cmake_minimum_required(VERSION 3.23)
project(MudaeBruteForce CUDA)

set(CMAKE_CUDA_STANDARD 14)

message("Filler message")

add_executable(MudaeBruteForce main.cu)

set_target_properties(MudaeBruteForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

