#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <chrono>  // Added for timing

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>  // Added for reduce
#include <thrust/system/cuda/execution_policy.h>

class SudokuBoards
{
public:
    uint32_t num_boards;
    uint32_t* repr; // Device pointer

    SudokuBoards(uint32_t n) : num_boards(n), repr(nullptr) {
        size_t bytes = 19ULL * n * sizeof(uint32_t);
        cudaError_t err = cudaMalloc(&repr, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        // No memset here; data will be copied or generated
    }

    // Delete copy constructor and assignment
    SudokuBoards(const SudokuBoards&) = delete;
    SudokuBoards& operator=(const SudokuBoards&) = delete;

    // Allow move semantics if needed
    SudokuBoards(SudokuBoards&& other) noexcept : num_boards(other.num_boards), repr(other.repr) {
        other.repr = nullptr;
        other.num_boards = 0;
    }

    SudokuBoards& operator=(SudokuBoards&& other) noexcept {
        if (this != &other) {
            if (repr != nullptr) {
                cudaFree(repr);
            }
            num_boards = other.num_boards;
            repr = other.repr;
            other.num_boards = 0;
            other.repr = nullptr;
        }
        return *this;
    }

    ~SudokuBoards() {
        if (repr != nullptr) {
            cudaFree(repr);
        }
    }

    uint32_t get_num_boards() const { return num_boards; }
};

__host__ __device__ inline void set_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    repr[repr_idx * num_boards + board_idx] |= (1U << bit_index);
}

__host__ __device__ inline void unset_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    repr[repr_idx * num_boards + board_idx] &= ~(1U << bit_index);
}

__host__ __device__ inline bool get_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    return ((repr[repr_idx * num_boards + board_idx] & (1U << bit_index)) > 0);
}

__host__ __device__ inline void set_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    uint8_t repr_idx = 17 - (pos >> 3); // end - (pos / 8)
    uint8_t bit_index = 28 - ((pos & 7) << 2); // 28 - (pos % 8) * 4
    repr[repr_idx * num_boards + board_idx] |= (number << bit_index);
}

__host__ __device__ inline void unset_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = 17 - (pos >> 3); // end - (pos / 8)
    uint8_t bit_index = 28 - ((pos & 7) << 2); // 28 - (pos % 8) * 4
    repr[repr_idx * num_boards + board_idx] &= (-1U ^ (15 << bit_index));
}

__host__ __device__ inline bool is_set(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = 17 - (pos >> 3); // end - (pos / 8)
    uint8_t bit_index = 28 - ((pos & 7) << 2); // 28 - (pos % 8) * 4
    return (repr[repr_idx * num_boards + board_idx] & (15 << bit_index)) > 0;
}

__host__ __device__ inline uint8_t get_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = 17 - (pos >> 3); // end - (pos / 8)
    uint8_t bit_index = 28 - ((pos & 7) << 2); // 28 - (pos % 8) * 4
    return (repr[repr_idx * num_boards + board_idx] >> bit_index) & 15;
}

__host__ __device__ inline bool is_blocked(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    return get_index(repr, num_boards, board_idx, 9 * (pos / 9) + number) ||
        get_index(repr, num_boards, board_idx, 81 + 9 * (pos % 9) + number) ||
        get_index(repr, num_boards, board_idx, 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27)) + number);
}

__host__ __device__ inline void set(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    set_index(repr, num_boards, board_idx, 9 * (pos / 9) + number);
    set_index(repr, num_boards, board_idx, 81 + 9 * (pos % 9) + number);
    set_index(repr, num_boards, board_idx, 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27)) + number);
    set_number_at_pos(repr, num_boards, board_idx, pos, number + 1);
}

__host__ __device__ inline void unset(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    unset_index(repr, num_boards, board_idx, 9 * (pos / 9) + number);
    unset_index(repr, num_boards, board_idx, 81 + 9 * (pos % 9) + number);
    unset_index(repr, num_boards, board_idx, 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27)) + number);
    unset_number_at_pos(repr, num_boards, board_idx, pos);
}

__host__ __device__ inline uint32_t get_id(uint32_t* repr, uint32_t num_boards, uint32_t board_idx) {
    return repr[18 * num_boards + board_idx];
}

const int MAX_LEVELS = 81;

// Kernel 1: Find Next Cell (MRV), return next_pos and number of boards to be generated
__global__ void find_next_cell_kernel(uint32_t* d_repr, uint32_t num_boards, uint8_t* d_next_pos, uint32_t* d_num_children_out) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= num_boards) return;

    int min_poss = 10;
    uint8_t best_pos = 0;
    bool is_solved = true;

    for (uint8_t pos = 0; pos < 81; ++pos) {
        if (!is_set(d_repr, num_boards, board_idx, pos)) {
            is_solved = false;
            int count = 0;
            for (uint8_t num = 0; num < 9; ++num) {
                if (!is_blocked(d_repr, num_boards, board_idx, pos, num)) {
                    ++count;
                }
            }
            if (count < min_poss) {
                min_poss = count;
                best_pos = pos;
            }
        }
    }

    uint32_t num_children;
    if (is_solved) {
        d_next_pos[board_idx] = 200;
        num_children = 1;
    }
    else if (min_poss == 0) {
        d_next_pos[board_idx] = 255;
        num_children = 0;
    }
    else {
        d_next_pos[board_idx] = best_pos;
        num_children = min_poss;
    }

    d_num_children_out[board_idx] = num_children;
}

// Kernel 2: Generate Children
__global__ void generate_children_kernel(uint32_t* d_in_repr, uint32_t in_num_boards, uint8_t* d_next_pos, uint32_t* d_prefixes, uint32_t* d_out_repr, uint32_t out_num_boards) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= in_num_boards) return;

    uint8_t pos = d_next_pos[board_idx];
    if (pos == 255) return; // impossible, no children

    uint32_t out_start = d_prefixes[board_idx];

    if (pos == 200) {
        // solved, copy as is
        uint32_t out_idx = out_start;
        for (uint8_t field = 0; field < 19; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        return;
    }

    // generate children
    uint32_t out_idx = out_start;
    for (uint8_t num = 0; num < 9; ++num) {
        if (!is_blocked(d_in_repr, in_num_boards, board_idx, pos, num)) {
            // copy board
            for (uint8_t field = 0; field < 19; ++field) {
                d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
            }
            // set the new value
            set(d_out_repr, out_num_boards, out_idx, pos, num);
            ++out_idx;
        }
    }
}

// Main function
std::vector<std::array<uint8_t, 81>> solve_multiple_sudoku(SudokuBoards* current) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    uint32_t original_num = current->get_num_boards();

    for (int level = 0; level < MAX_LEVELS; ++level) {
        uint32_t num_boards = current->get_num_boards();
        std::cout << "Level " << level << ", Boards: " << num_boards << std::endl;
        if (num_boards == 0) break;

        uint8_t* d_next_pos = nullptr;
        err = cudaMalloc(&d_next_pos, num_boards * sizeof(uint8_t));
        if (err != cudaSuccess) { /* Handle error */ }

        uint32_t* d_num_children_out = nullptr;
        err = cudaMalloc(&d_num_children_out, num_boards * sizeof(uint32_t));
        if (err != cudaSuccess) { /* Handle error */ }

        int threads = 256;
        int blocks = (num_boards + threads - 1) / threads;
        find_next_cell_kernel << <blocks, threads, 0, stream >> > (current->repr, num_boards, d_next_pos, d_num_children_out);
        cudaStreamSynchronize(stream);

        // Compute new_num using Thrust reduce on device
        uint32_t new_num = thrust::reduce(thrust::cuda::par.on(stream),
            thrust::device_ptr<uint32_t>(d_num_children_out),
            thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
            0u);

        if (new_num == 0) {
            cudaFree(d_next_pos);
            cudaFree(d_num_children_out);
            break;
        }

        uint32_t* d_prefixes = nullptr;
        err = cudaMalloc(&d_prefixes, num_boards * sizeof(uint32_t));
        if (err != cudaSuccess) { /* Handle error */ }

        thrust::exclusive_scan(thrust::cuda::par.on(stream),
            thrust::device_ptr<uint32_t>(d_num_children_out),
            thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
            thrust::device_ptr<uint32_t>(d_prefixes));

        SudokuBoards* out_ptr = new SudokuBoards(new_num);

        generate_children_kernel << <blocks, threads, 0, stream >> > (current->repr, num_boards, d_next_pos, d_prefixes, out_ptr->repr, new_num);
        cudaStreamSynchronize(stream);

        cudaFree(d_next_pos);
        cudaFree(d_num_children_out);
        cudaFree(d_prefixes);

        // Clean up old current
        delete current;

        // Move to new
        current = out_ptr;
    }

    uint32_t final_num_boards = current->get_num_boards();
    size_t bytes = 19ULL * final_num_boards * sizeof(uint32_t);
    uint32_t* h_repr = (uint32_t*)malloc(bytes);
    cudaMemcpyAsync(h_repr, current->repr, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::array<uint8_t, 81>> solutions(original_num);
    std::vector<bool> found(original_num, false);

    for (uint32_t board_idx = 0; board_idx < final_num_boards; ++board_idx) {
        uint32_t id = get_id(h_repr, final_num_boards, board_idx);
        if (id >= original_num || found[id]) continue;

        auto& sol = solutions[id];
        bool valid = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t num = get_number_at_pos(h_repr, final_num_boards, board_idx, pos) - 1;
            if (num + 1 == 0) {  // Check for unset (0)
                valid = false;
                break;
            }
            sol[pos] = num + 1;
        }
        if (valid) {
            found[id] = true;
        }
    }

    free(h_repr);

    // Final cleanup
    delete current;
    cudaStreamDestroy(stream);

    return solutions;
}

void solveGPU(const std::string& input_file, const std::string& output_file, int count) {
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_file << std::endl;
        return;
    }

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    std::string line;
    std::vector<std::vector<uint32_t>> temp_reprs;
    int id = 0;
    while (std::getline(file, line) && id < count) {
        if (line.length() != 81) continue;

        std::vector<uint32_t> h_single_repr(19, 0);
        bool valid = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            char c = line[pos];
            if (c < '0' || c > '9') {
                valid = false;
                break;
            }
            if (c != '0') {
                uint8_t number = c - '0' - 1;
                if (is_blocked(h_single_repr.data(), 1, 0, pos, number)) {
                    valid = false;
                    break;
                }
                set(h_single_repr.data(), 1, 0, pos, number);
            }
        }
        if (valid) {
            temp_reprs.push_back(std::move(h_single_repr));
            ++id;
        }
    }
    file.close();

    uint32_t num_boards = temp_reprs.size();
    if (num_boards == 0) {
        cudaStreamDestroy(stream);
        return;
    }

    uint32_t* h_repr = new uint32_t[19ULL * num_boards]();
    for (uint32_t b = 0; b < num_boards; ++b) {
        for (uint8_t f = 0; f < 18; ++f) {
            h_repr[f * num_boards + b] = temp_reprs[b][f];
        }
        h_repr[18 * num_boards + b] = b;
    }
    temp_reprs.clear();

    // Start timing after reading
    auto start = std::chrono::high_resolution_clock::now();

    SudokuBoards* inputs_ptr = new SudokuBoards(num_boards);
    size_t bytes = 19ULL * num_boards * sizeof(uint32_t);
    cudaMemcpyAsync(inputs_ptr->repr, h_repr, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    delete[] h_repr;

    std::vector<std::array<uint8_t, 81>> solutions = solve_multiple_sudoku(inputs_ptr);

    // End timing before writing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    cudaStreamDestroy(stream);

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }
    for (const auto& solution : solutions) {
        bool is_valid_solution = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t num = solution[pos];
            if (num == 0) {
                is_valid_solution = false;
                break;
            }
        }
        if (!is_valid_solution) {
            out << "No solution" << std::endl;
            continue;
        }
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t num = solution[pos];
            out << static_cast<char>('0' + num);
        }
        out << std::endl;
    }
    out.close();
}