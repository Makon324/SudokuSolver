cmake_minimum_required(VERSION 3.18)
project(sudoku LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 35)

add_executable(sudoku main.cpp CPUSolver.cpp GPUSolver.cu)
set_target_properties(sudoku PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable warnings for C++ and CUDA, with suppression for -Wreorder in CUDA
target_compile_options(sudoku PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets --expt-extended-lambda -Xcompiler=-Wall,-Wextra,-Wno-reorder --Wdefault-stream-launch --Wext-lambda-captures-this>
)