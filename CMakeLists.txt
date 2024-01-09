cmake_minimum_required(VERSION 3.23)

# I expect both of these variables to be blank before I set them
# but I print before I set incase it changes and I need to look into what's causing that
message("CUDA ARCHS: " ${CMAKE_CUDA_ARCHITECTURES})
set(CMAKE_CUDA_ARCHITECTURES 75)
message("CUDA ARCHS: " ${CMAKE_CUDA_ARCHITECTURES})

message("CUDA PATH: " ${CUDA_PATH})
file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDA_PATH)
message("CUDA PATH: " ${CUDA_PATH})

# Initialize project
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

# Add CMake Includes and Libraries to VS Solution: https://stackoverflow.com/a/8879306
# and use target_includes_directories instead of includes_directories: https://stackoverflow.com/a/13703725
set(CUDA_INCLUDES_DIR ${CUDA_PATH}/include)
message("CUDA_INCLUDES_DIR: " ${CUDA_INCLUDES_DIR})
set(CUDA_HEADER_FILES ${CUDA_INCLUDES_DIR}/cuda_runtime.h)
message("CUDA_HEADER_FILES: " ${CUDA_HEADER_FILES})

set(SOURCES
    main.cu
    read_sets.cu
    constants.cu
)

add_executable(MudaeBruteForce ${SOURCES} ${CUDA_HEADER_FILES})
# add_executable(MudaeBruteForce ${SOURCES} "${CUDA_HEADER_FILES}")
target_include_directories(MudaeBruteForce PRIVATE "${CUDA_INCLUDES_DIR}")

set_target_properties(MudaeBruteForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)