cmake_minimum_required(VERSION 3.23)
project(MudaeBruteForce CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Shut up about the "Integer conversion resulted in a change of sign"
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 68")
message(Flags: ${CMAKE_CUDA_FLAGS})

message("Filler message")

add_executable(MudaeBruteForce main.cu)

set_target_properties(MudaeBruteForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

