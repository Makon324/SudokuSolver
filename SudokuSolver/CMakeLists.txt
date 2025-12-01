cmake_minimum_required(VERSION 3.18)
project(sudoku CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 35)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")

add_executable(sudoku GPUSolver.cu)

target_link_libraries(sudoku ${CUDA_LIBRARIES})