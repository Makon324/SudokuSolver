cmake_minimum_required(VERSION 3.16)

project(SudokuSolver LANGUAGES CXX CUDA)

# Set CUDA properties (adjust standard as needed for your code)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60;70;80;86;89;90")  # Common modern GPU arches; adjust for your node (e.g., based on CUDA 11.4 compatibility)

# Find CUDA toolkit (handles paths automatically on Linux)
find_package(CUDAToolkit REQUIRED)

# No need for VS-specific architecture check or fatal error

# Global configuration types (keep these, but no MSVC flag overrides needed on Linux)

# Remove all MSVC-specific blocks, as they interfere on Linux

# Nuget packages function stub (keep if used, but it's a warning stub; remove if not needed)
function(use_package TARGET PACKAGE VERSION)
    message(WARNING "No implementation of use_package. Create yours. "
                    "Package \"${PACKAGE}\" with version \"${VERSION}\" "
                    "for target \"${TARGET}\" is ignored!")
endfunction()

# Common utils (include only if it doesn't cause issues; on Linux, use_props from here is VS-specific and should be commented out in subprojects)
include(CMake/Utils.cmake OPTIONAL)

# Use solution folders feature (optional, harmless on Linux)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sub-projects
add_subdirectory(SudokuSolver)

# Additional notes to fully fix your project:
# 1. In SudokuSolver/CMakeLists.txt, comment out or remove the lines calling 'use_props("/BuildCustomizations/CUDA 13.0.cmake")' – that's VS/Windows-only and causes the warnings.
# 2. In SudokuSolver/CMakeLists.txt, for target_link_libraries(SudokuSolver ...), remove all Windows libs (kernel32, user32, gdi32, winspool, comdlg32, advapi32, shell32, ole32, oleaut32, uuid, odbc32, odbccp32).
#    Replace '-lcudart_static' with 'CUDA::cudart_static' (modern CMake handles it).
#    Full example: target_link_libraries(SudokuSolver PRIVATE CUDA::cudart_static)
# 3. If your subproject has .cu files, add them with set_source_files_properties(... PROPERTIES LANGUAGE CUDA)
# 4. After fixes, run: cd ~/SudokuSolver/build && rm -rf * && cmake .. && make -j$(nproc)