cmake_minimum_required(VERSION 3.23)

message("CUDA ARCHS: " ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_ARCHITECTURES 75)
message("CUDA ARCHS: " ${CMAKE_CUDA_ARCHITECTURES})

project(MudaeBruteForce CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Shut up about the "Integer conversion resulted in a change of sign"
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 68")
message(Flags: ${CMAKE_CUDA_FLAGS})

# message("Filler message")

add_compile_definitions(MAX_DL=50)
add_compile_definitions(MAX_FREE_BUNDLES=5)
add_compile_definitions(OVERLAP_LIMIT=30000)
add_compile_definitions(DISABLED_SETS_SIZE=MAX_DL+MAX_FREE_BUNDLES)

add_executable(MudaeBruteForce main.cu)

set_target_properties(MudaeBruteForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

