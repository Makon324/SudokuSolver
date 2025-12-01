#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


class SudokuBoards
{
    uint32_t num_boards;
    __device__ __host__ inline void set_index(uint32_t board_idx, uint16_t index) {
        uint8_t repr_idx = index >> 5;
        uint8_t bit_index = index & 31;
        repr[repr_idx * num_boards + board_idx] |= (1U << bit_index);
    }
    __device__ __host__ inline void unset_index(uint32_t board_idx, uint16_t index) {
        uint8_t repr_idx = index >> 5;
        uint8_t bit_index = index & 31;
        repr[repr_idx * num_boards + board_idx] &= ~(1U << bit_index);
    }
    __device__ __host__ inline bool get_index(uint32_t board_idx, uint16_t index) {
        uint8_t repr_idx = index >> 5;
        uint8_t bit_index = index & 31;
        return ((repr[repr_idx * num_boards + board_idx] & (1U << bit_index)) > 0);
    }
    __device__ __host__ inline void set_number_at_pos(uint32_t board_idx, uint8_t pos, uint8_t number) {
        uint8_t repr_idx = 17 - (pos >> 3);
        uint8_t bit_index = 28 - ((pos & 7) << 2);
        repr[repr_idx * num_boards + board_idx] |= (number << bit_index);
    }
    __device__ __host__ inline void unset_number_at_pos(uint32_t board_idx, uint8_t pos) {
        uint8_t repr_idx = 17 - (pos >> 3);
        uint8_t bit_index = 28 - ((pos & 7) << 2);
        repr[repr_idx * num_boards + board_idx] &= (-1U ^ (15 << bit_index));
    }
public:
    uint32_t* repr;
    __device__ __host__ uint32_t get_num_boards() const { return num_boards; }
    // Constructor: allocate unified memory
    SudokuBoards(uint32_t n) : num_boards(n) {
        size_t bytes = 19ULL * n * sizeof(uint32_t);
        cudaError_t err = cudaMallocManaged(&repr, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaMemset(repr, 0, bytes);
        // Prefetch handled outside
    }
    // Delete copy
    SudokuBoards(const SudokuBoards&) = delete;
    SudokuBoards& operator=(const SudokuBoards&) = delete;
    // Move semantics
    SudokuBoards(SudokuBoards&& other) noexcept : num_boards(other.num_boards), repr(other.repr) {
        other.repr = nullptr;
        other.num_boards = 0;
    }
    SudokuBoards& operator=(SudokuBoards&& other) noexcept {
        if (this != &other) {
            if (repr) cudaFree(repr);
            num_boards = other.num_boards;
            repr = other.repr;
            other.repr = nullptr;
            other.num_boards = 0;
        }
        return *this;
    }
    ~SudokuBoards() {
        if (repr != nullptr) {
            cudaFree(repr);
        }
    }
    __device__ __host__ uint32_t get_id(uint32_t board_idx) const {
        return repr[18 * num_boards + board_idx];
    }
    __device__ __host__ void set(uint32_t board_idx, uint8_t pos, uint8_t number) {
        set_index(board_idx, 9 * (pos / 9) + number);
        set_index(board_idx, 81 + 9 * (pos % 9) + number);
        set_index(board_idx, 162 + 9 * ((pos / 27) * 3 + ((pos % 9) / 3)) + number);
        set_number_at_pos(board_idx, pos, number + 1);
    }
    __device__ __host__ bool is_set(uint32_t board_idx, uint8_t pos) {
        uint8_t repr_idx = 17 - (pos >> 3);
        uint8_t bit_index = 28 - ((pos & 7) << 2);
        return (repr[repr_idx * num_boards + board_idx] & (15 << bit_index)) > 0;
    }
    __device__ __host__ void unset(uint32_t board_idx, uint8_t pos, uint8_t number) {
        unset_index(board_idx, 9 * (pos / 9) + number);
        unset_index(board_idx, 81 + 9 * (pos % 9) + number);
        unset_index(board_idx, 162 + 9 * ((pos / 27) * 3 + ((pos % 9) / 3)) + number);
        unset_number_at_pos(board_idx, pos);
    }
    __device__ __host__ bool is_blocked(uint32_t board_idx, uint8_t pos, uint8_t number) {
        return get_index(board_idx, 9 * (pos / 9) + number) ||
            get_index(board_idx, 81 + 9 * (pos % 9) + number) ||
            get_index(board_idx, 162 + 9 * ((pos / 27) * 3 + ((pos % 9) / 3)) + number);
    }
    __device__ __host__ uint8_t get_number_at_pos(uint32_t board_idx, uint8_t pos) {
        uint8_t repr_idx = 17 - (pos >> 3);
        uint8_t bit_index = 28 - ((pos & 7) << 2);
        return (repr[repr_idx * num_boards + board_idx] >> bit_index) & 15;
    }
};
const int MAX_LEVELS = 81;
__global__ void find_next_cell_kernel(SudokuBoards& boards, uint8_t* next_pos, uint32_t* num_children_out) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= boards.get_num_boards()) return;
    int min_poss = 10;
    uint8_t best_pos = 0;
    bool is_solved = true;
    for (uint8_t pos = 0; pos < 81; ++pos) {
        if (!boards.is_set(board_idx, pos)) {
            is_solved = false;
            int count = 0;
            for (uint8_t num = 0; num < 9; ++num) {
                if (!boards.is_blocked(board_idx, pos, num)) ++count;
            }
            if (count < min_poss) {
                min_poss = count;
                best_pos = pos;
            }
        }
    }
    uint32_t num_children;
    if (is_solved) {
        next_pos[board_idx] = 200;
        num_children = 1;
    }
    else if (min_poss == 0) {
        next_pos[board_idx] = 255;
        num_children = 0;
    }
    else {
        next_pos[board_idx] = best_pos;
        num_children = min_poss;
    }
    num_children_out[board_idx] = num_children;
}
__global__ void generate_children_kernel(SudokuBoards& in_boards, uint8_t* next_pos, uint32_t* prefixes, SudokuBoards& out_boards) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= in_boards.get_num_boards()) return;
    uint8_t pos = next_pos[board_idx];
    if (pos == 255) return;
    uint32_t out_start = prefixes[board_idx];
    if (pos == 200) {
        uint32_t out_idx = out_start;
        for (uint8_t field = 0; field < 19; ++field) {
            out_boards.repr[field * out_boards.get_num_boards() + out_idx] =
                in_boards.repr[field * in_boards.get_num_boards() + board_idx];
        }
        return;
    }
    uint32_t out_idx = out_start;
    for (uint8_t num = 0; num < 9; ++num) {
        if (!in_boards.is_blocked(board_idx, pos, num)) {
            for (uint8_t field = 0; field < 19; ++field) {
                out_boards.repr[field * out_boards.get_num_boards() + out_idx] =
                    in_boards.repr[field * in_boards.get_num_boards() + board_idx];
            }
            out_boards.set(out_idx, pos, num);
            ++out_idx;
        }
    }
}
std::vector<std::array<uint8_t, 81>> solve_multiple_sudoku(SudokuBoards& inputs, bool supports_prefetch) {
    uint32_t original_num = inputs.get_num_boards();
    SudokuBoards current(std::move(inputs));
    for (int level = 0; level < MAX_LEVELS; ++level) {
        uint32_t num_boards = current.get_num_boards();
        std::cout << "Level " << level << ", boards: " << num_boards << std::endl;
        if (num_boards == 0) break;
        uint8_t* next_pos = nullptr;
        uint32_t* num_children_out = nullptr;
        cudaError_t err = cudaMallocManaged(&next_pos, num_boards * sizeof(uint8_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged (next_pos) failed: %s\n", cudaGetErrorString(err));
            break;
        }
        err = cudaMallocManaged(&num_children_out, num_boards * sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged (num_children_out) failed: %s\n", cudaGetErrorString(err));
            break;
        }
        // Prefetch auxiliaries to GPU if supported
        if (supports_prefetch) {
            err = cudaMemPrefetchAsync(next_pos, num_boards * sizeof(uint8_t), 0, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync (next_pos) to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
            err = cudaMemPrefetchAsync(num_children_out, num_boards * sizeof(uint32_t), 0, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync (num_children_out) to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
        }
        int threads = 128;
        int blocks = (num_boards + threads - 1) / threads;
        find_next_cell_kernel << <blocks, threads >> > (current, next_pos, num_children_out);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "find_next_cell_kernel failed: %s\n", cudaGetErrorString(err));
            // Handle error as needed, e.g., break or exit
        }
        // Prefetch num_children_out back to CPU if supported
        if (supports_prefetch) {
            err = cudaMemPrefetchAsync(num_children_out, num_boards * sizeof(uint32_t), cudaCpuDeviceId, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync (num_children_out) to CPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
        }
        // Compute new_num and prefixes on CPU
        uint32_t* prefixes = nullptr;
        err = cudaMallocManaged(&prefixes, num_boards * sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged (prefixes) failed: %s\n", cudaGetErrorString(err));
            break;
        }
        if (supports_prefetch) {
            err = cudaMemPrefetchAsync(prefixes, num_boards * sizeof(uint32_t), 0, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync (prefixes) to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
        }
        uint32_t new_num = 0;
        for (uint32_t i = 0; i < num_boards; ++i) {
            prefixes[i] = new_num;
            new_num += num_children_out[i];
        }
        if (new_num == 0) {
            cudaFree(next_pos); cudaFree(num_children_out);
            cudaFree(prefixes);
            break;
        }
        SudokuBoards out_boards(new_num);
        generate_children_kernel << <blocks, threads >> > (current, next_pos, prefixes, out_boards);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "generate_children_kernel failed: %s\n", cudaGetErrorString(err));
            // Handle error
        }
        cudaFree(next_pos);
        cudaFree(num_children_out);
        cudaFree(prefixes);
        current = std::move(out_boards);
    }
    // Prefetch final results to CPU if supported
    size_t bytes = 19ULL * current.get_num_boards() * sizeof(uint32_t);
    cudaError_t err;
    if (supports_prefetch) {
        err = cudaMemPrefetchAsync(current.repr, bytes, cudaCpuDeviceId, nullptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Final prefetch to CPU failed: %s (continuing)\n", cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();
    std::vector<std::array<uint8_t, 81>> solutions(original_num);
    std::vector<bool> found(original_num, false);
    for (uint32_t board_idx = 0; board_idx < current.get_num_boards(); ++board_idx) {
        uint32_t id = current.get_id(board_idx);
        if (id >= original_num || found[id]) continue;
        auto& sol = solutions[id];
        bool valid = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t num = current.get_number_at_pos(board_idx, pos);
            if (num == 0) { valid = false; break; }
            sol[pos] = num;
        }
        if (valid) found[id] = true;
    }
    return solutions;
}
void solveGPU(const std::string& input_file, const std::string& output_file, int count) {
    // Check for available GPUs with error checking
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Cannot run GPU solver." << std::endl;
        return;
    }
    // Set to device 0 with error checking
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // Optional reset
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Warning: cudaDeviceReset failed: " << cudaGetErrorString(err) << " (continuing)" << std::endl;
    }
    // Get device properties to diagnose CC
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "Using device: " << prop.name << ", Compute capability: " << prop.major << "." << prop.minor << std::endl;

    bool supports_prefetch = (prop.major >= 6);

    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_file << std::endl;
        return;
    }
    std::string line;
    std::vector<SudokuBoards> temp_boards;
    int id = 0;
    while (std::getline(file, line) && id < count) {
        if (line.length() != 81) continue;
        SudokuBoards temp(1);
        // Prefetch if supported
        if (supports_prefetch) {
            size_t bytes = 19 * sizeof(uint32_t);
            err = cudaMemPrefetchAsync(temp.repr, bytes, 0, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
        }
        bool valid = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            char c = line[pos];
            if (c < '0' || c > '9') { valid = false; break; }
            if (c != '0') {
                uint8_t number = c - '0' - 1;
                if (temp.is_blocked(0, pos, number)) { valid = false; break; }
                temp.set(0, pos, number);
            }
        }
        if (valid) {
            temp.repr[18] = id; // set ID
            temp_boards.push_back(std::move(temp));
            ++id;
        }
    }
    file.close();
    uint32_t num_boards = temp_boards.size();
    if (num_boards == 0) return;
    SudokuBoards inputs(num_boards);
    // Prefetch inputs if supported
    if (supports_prefetch) {
        size_t bytes = 19ULL * num_boards * sizeof(uint32_t);
        err = cudaMemPrefetchAsync(inputs.repr, bytes, 0, nullptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: cudaMemPrefetchAsync to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
        }
    }
    for (uint32_t b = 0; b < num_boards; ++b) {
        for (uint8_t f = 0; f < 19; ++f) {
            inputs.repr[f * num_boards + b] = temp_boards[b].repr[f];
        }
    }
    std::vector<std::array<uint8_t, 81>> solutions = solve_multiple_sudoku(inputs, supports_prefetch);
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }
    for (const auto& solution : solutions) {
        bool has_solution = true;
        for (uint8_t n : solution) {
            if (n == 0) { has_solution = false; break; }
        }
        if (!has_solution) {
            out << "No solution" << std::endl;
            continue;
        }
        for (uint8_t n : solution)
            out << static_cast<char>('0' + n);
        out << std::endl;
    }
    out.close();
}