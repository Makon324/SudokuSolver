cmake_minimum_required(VERSION 3.18)
project(sudoku LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 35)

# Suppress deprecated GPU targets warning
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(sudoku main.cpp CPUSolver.cpp GPUSolver.cu)
set_target_properties(sudoku PROPERTIES CUDA_SEPARABLE_COMPILATION ON)