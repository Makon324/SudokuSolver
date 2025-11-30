cmake_minimum_required(VERSION 3.18)

project(SudokuSolver LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(sudoku
  main.cpp
  CPUSolver.cpp
  GPUSolver.cu
)

target_link_libraries(sudoku
  ${CUDA_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
)