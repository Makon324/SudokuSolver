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
        // Check if device supports concurrent managed access before prefetch
        int concurrentMA = 0;
        cudaDeviceGetAttribute(&concurrentMA, cudaDevAttrConcurrentManagedAccess, 0);
        if (concurrentMA) {
            // Prefetch to device 0 (assuming single GPU)
            err = cudaMemPrefetchAsync(repr, bytes, 0, nullptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: cudaMemPrefetchAsync to GPU failed: %s (continuing)\n", cudaGetErrorString(err));
            }
        } // else skip prefetch, as not supported on this device
        // Hint: CPU will only read at the end
        //err = cudaMemAdvise(repr, bytes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
        //if (err != cudaSuccess) {
        //    fprintf(stderr, "Warning: cudaMemAdvise failed: %s\n", cudaGetErrorString(err));
        //}
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
    // Removed: cuda::atomic<uint32_t>& total = *reinterpret_cast<cuda::atomic<uint32_t>*>(total_new);
    // Removed: total.fetch_add(num_children);
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
std::vector<std::array<uint8_t, 81>> solve_multiple_sudoku(SudokuBoards& inputs) {
    uint32_t original_num = inputs.get_num_boards();
    SudokuBoards current(std::move(inputs));
    for (int level = 0; level < MAX_LEVELS; ++level) {
        uint32_t num_boards = current.get_num_boards();
		std::cout << "Level " << level << ", number of boards: " << num_boards << std::endl;
        if (num_boards == 0) break;
        uint8_t* next_pos = nullptr;
        uint32_t* num_children_out = nullptr;
        cudaMallocManaged(&next_pos, num_boards * sizeof(uint8_t));
        cudaMallocManaged(&num_children_out, num_boards * sizeof(uint32_t));
        int threads = 128;
        int blocks = (num_boards + threads - 1) / threads;
        find_next_cell_kernel << <blocks, threads >> > (current, next_pos, num_children_out);
        cudaDeviceSynchronize();
        // New: Compute new_num and prefixes on CPU
        uint32_t* prefixes = nullptr;
        cudaMallocManaged(&prefixes, num_boards * sizeof(uint32_t));
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
        cudaFree(next_pos);
        cudaFree(num_children_out);
        cudaFree(prefixes);
        current = std::move(out_boards);
    }
    // Bring final results to CPU (prefetch commented out to avoid warning; data migrates on access)
    // size_t bytes = 19ULL * current.get_num_boards() * sizeof(uint32_t);
    // cudaError_t err = cudaMemPrefetchAsync(current.repr, bytes, cudaCpuDeviceId, nullptr);
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Final prefetch to CPU failed: %s (continuing)\n", cudaGetErrorString(err));
    // }
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
    // Check for available GPUs
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Cannot run GPU solver." << std::endl;
        return;
    }
    // Set to device 0
    cudaSetDevice(0);
    cudaDeviceReset(); // optional: clean state

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
    for (uint32_t b = 0; b < num_boards; ++b) {
        for (uint8_t f = 0; f < 19; ++f) {
            inputs.repr[f * num_boards + b] = temp_boards[b].repr[f];
        }
    }
    std::vector<std::array<uint8_t, 81>> solutions = solve_multiple_sudoku(inputs);
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