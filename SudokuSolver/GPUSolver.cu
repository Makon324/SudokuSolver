#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/system/cuda/execution_policy.h>

#ifdef __INTELLISENSE__
#include <intrin.h>
int __popc(unsigned int x) { return __popcnt(x); }
int __ffs(unsigned int x) {
    unsigned long index;
    if (x == 0) return 0;
    _BitScanForward(&index, x);
    return index + 1;
}
#endif

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

class SudokuBoards
{
public:
    uint32_t num_boards;
    uint32_t* repr; // Device pointer

    SudokuBoards(uint32_t n) : num_boards(n), repr(nullptr) {
        size_t bytes = 19ULL * n * sizeof(uint32_t);
        cudaError_t err = cudaMalloc(&repr, bytes);
        checkCudaError(err, "cudaMalloc failed in SudokuBoards constructor");
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
                checkCudaError(cudaFree(repr), "cudaFree failed in SudokuBoards move assignment");
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
            checkCudaError(cudaFree(repr), "cudaFree failed in SudokuBoards destructor");
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

__device__ uint32_t get_mask(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t base_index) {
    uint8_t repr_idx = base_index >> 5;
    uint8_t bit_index = base_index & 31;
    uint32_t val1 = repr[repr_idx * num_boards + board_idx];
    if (bit_index <= 23) {
        return (val1 >> bit_index) & 0x1FF;
    }
    else {
        uint32_t val2 = repr[(repr_idx + 1) * num_boards + board_idx];
        uint32_t low = val1 >> bit_index;
        uint32_t high = val2 << (32 - bit_index);
        return (low | high) & 0x1FF;
    }
}

const int MAX_LEVELS = 81;

// Kernel 1: Find Next Cell (MRV), return next_pos and number of boards to be generated
__global__ void find_next_cell_kernel(uint32_t* d_repr, uint32_t num_boards, uint8_t* d_next_pos, uint32_t* d_num_children_out) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= num_boards) return;

    // Propagation of singles
    bool changed = true;
    bool impossible = false;
    while (changed && !impossible) {
        changed = false;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            if (is_set(d_repr, num_boards, board_idx, pos)) continue;
            uint16_t base_row = 9 * (pos / 9);
            uint16_t base_col = 81 + 9 * (pos % 9);
            uint16_t base_box = 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27));
            uint32_t row_m = get_mask(d_repr, num_boards, board_idx, base_row);
            uint32_t col_m = get_mask(d_repr, num_boards, board_idx, base_col);
            uint32_t box_m = get_mask(d_repr, num_boards, board_idx, base_box);
            uint32_t used = row_m | col_m | box_m;
            uint32_t avail = ~used & 0x1FF;
            int count = __popc(avail);
            if (count == 0) {
                impossible = true;
                break;
            }
            else if (count == 1) {
                uint32_t bit = __ffs(avail) - 1;
                set(d_repr, num_boards, board_idx, pos, bit);
                changed = true;
            }
        }
    }

    if (impossible) {
        d_next_pos[board_idx] = 255;
        d_num_children_out[board_idx] = 0;
        return;
    }

    // Now find MRV
    int min_poss = 10;
    uint8_t best_pos = 0;
    bool is_solved = true;

    for (uint8_t pos = 0; pos < 81; ++pos) {
        if (!is_set(d_repr, num_boards, board_idx, pos)) {
            is_solved = false;
            uint16_t base_row = 9 * (pos / 9);
            uint16_t base_col = 81 + 9 * (pos % 9);
            uint16_t base_box = 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27));
            uint32_t row_m = get_mask(d_repr, num_boards, board_idx, base_row);
            uint32_t col_m = get_mask(d_repr, num_boards, board_idx, base_col);
            uint32_t box_m = get_mask(d_repr, num_boards, board_idx, base_box);
            uint32_t used = row_m | col_m | box_m;
            uint32_t avail = ~used & 0x1FF;
            int count = __popc(avail);
            if (count == 0) {
                impossible = true;
                break;
            }
            if (count < min_poss) {
                min_poss = count;
                best_pos = pos;
            }
        }
    }

    uint32_t num_children;
    if (impossible) {
        d_next_pos[board_idx] = 255;
        num_children = 0;
    }
    else if (is_solved) {
        d_next_pos[board_idx] = 200;
        num_children = 1;
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
    uint16_t base_row = 9 * (pos / 9);
    uint16_t base_col = 81 + 9 * (pos % 9);
    uint16_t base_box = 162 + 9 * (((pos % 9) / 3) * 3 + (pos / 27));
    uint32_t row_m = get_mask(d_in_repr, in_num_boards, board_idx, base_row);
    uint32_t col_m = get_mask(d_in_repr, in_num_boards, board_idx, base_col);
    uint32_t box_m = get_mask(d_in_repr, in_num_boards, board_idx, base_box);
    uint32_t used = row_m | col_m | box_m;
    uint32_t avail = ~used & 0x1FF;
    uint32_t out_idx = out_start;
    while (avail != 0) {
        uint32_t bit = __ffs(avail) - 1;
        for (uint8_t field = 0; field < 19; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        set(d_out_repr, out_num_boards, out_idx, pos, bit);
        avail &= ~(1u << bit);
        ++out_idx;
    }
}

// Main function
std::vector<std::array<uint8_t, 81>> solve_multiple_sudoku(SudokuBoards* current) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    checkCudaError(err, "cudaStreamCreate failed in solve_multiple_sudoku");

    uint32_t original_num = current->get_num_boards();

    const uint32_t MAX_PREALLOC_BOARDS = 1048576; // 1M boards, adjust based on expected max
    size_t prealloc_bytes = 19ULL * MAX_PREALLOC_BOARDS * sizeof(uint32_t);

    uint32_t* repr_a = nullptr;
    checkCudaError(cudaMalloc(&repr_a, prealloc_bytes), "cudaMalloc failed for repr_a");

    uint32_t* repr_b = nullptr;
    checkCudaError(cudaMalloc(&repr_b, prealloc_bytes), "cudaMalloc failed for repr_b");

    // Assume original_num <= MAX_PREALLOC_BOARDS; if not, handle similarly by allocating larger
    size_t initial_bytes = 19ULL * original_num * sizeof(uint32_t);
    checkCudaError(cudaMemcpy(repr_a, current->repr, initial_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy failed for initial copy");
    delete current;

    uint32_t* input_repr = repr_a;
    uint32_t input_max = MAX_PREALLOC_BOARDS;
    uint32_t* output_repr = repr_b;
    uint32_t output_max = MAX_PREALLOC_BOARDS;
    uint32_t num_boards = original_num;

    for (int loop = 0; loop < MAX_LEVELS; ++loop) {
        std::cout << "Loop " << loop << ", Boards: " << num_boards << std::endl;
        if (num_boards == 0) break;

        uint8_t* d_next_pos = nullptr;
        err = cudaMalloc(&d_next_pos, num_boards * sizeof(uint8_t));
        checkCudaError(err, "cudaMalloc failed for d_next_pos");

        uint32_t* d_num_children_out = nullptr;
        err = cudaMalloc(&d_num_children_out, num_boards * sizeof(uint32_t));
        checkCudaError(err, "cudaMalloc failed for d_num_children_out");

        int threads = 256;
        int blocks = (num_boards + threads - 1) / threads;
        find_next_cell_kernel << <blocks, threads, 0, stream >> > (input_repr, num_boards, d_next_pos, d_num_children_out);
        checkCudaError(cudaGetLastError(), "find_next_cell_kernel launch failed");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after find_next_cell_kernel");

        // Compute new_num using Thrust reduce on device
        uint32_t new_num = thrust::reduce(thrust::cuda::par.on(stream),
            thrust::device_ptr<uint32_t>(d_num_children_out),
            thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
            0u);
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after thrust::reduce");

        if (new_num == 0) {
            checkCudaError(cudaFree(d_next_pos), "cudaFree failed for d_next_pos");
            checkCudaError(cudaFree(d_num_children_out), "cudaFree failed for d_num_children_out");
            break;
        }

        // Compute unsolved_count using Thrust count_if
        uint32_t unsolved_count = thrust::count_if(thrust::cuda::par.on(stream),
            thrust::device_ptr<uint8_t>(d_next_pos),
            thrust::device_ptr<uint8_t>(d_next_pos + num_boards),
            [] __device__(uint8_t pos) { return pos < 200; });
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after thrust::count_if");

        bool all_solved = (unsolved_count == 0);

        uint32_t* d_prefixes = nullptr;
        err = cudaMalloc(&d_prefixes, num_boards * sizeof(uint32_t));
        checkCudaError(err, "cudaMalloc failed for d_prefixes");

        thrust::exclusive_scan(thrust::cuda::par.on(stream),
            thrust::device_ptr<uint32_t>(d_num_children_out),
            thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
            thrust::device_ptr<uint32_t>(d_prefixes));
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after thrust::exclusive_scan");

        // Handle output buffer, reallocate if overflow
        if (new_num > output_max) {
            checkCudaError(cudaFree(output_repr), "cudaFree failed for output_repr (overflow)");
            size_t new_bytes = 19ULL * new_num * sizeof(uint32_t);
            checkCudaError(cudaMalloc(&output_repr, new_bytes), "cudaMalloc failed for output_repr (overflow)");
            output_max = new_num;
        }

        generate_children_kernel << <blocks, threads, 0, stream >> > (input_repr, num_boards, d_next_pos, d_prefixes, output_repr, new_num);
        checkCudaError(cudaGetLastError(), "generate_children_kernel launch failed");
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after generate_children_kernel");

        checkCudaError(cudaFree(d_next_pos), "cudaFree failed for d_next_pos");
        checkCudaError(cudaFree(d_num_children_out), "cudaFree failed for d_num_children_out");
        checkCudaError(cudaFree(d_prefixes), "cudaFree failed for d_prefixes");

        // Swap input and output
        std::swap(input_repr, output_repr);
        std::swap(input_max, output_max);
        num_boards = new_num;

        if (all_solved) {
            break;
        }
    }

    uint32_t final_num_boards = num_boards;
    size_t bytes = 19ULL * final_num_boards * sizeof(uint32_t);
    uint32_t* h_repr = (uint32_t*)malloc(bytes);
    if (h_repr == nullptr) {
        fprintf(stderr, "malloc failed for h_repr\n");
        exit(1);
    }
    err = cudaMemcpyAsync(h_repr, input_repr, bytes, cudaMemcpyDeviceToHost, stream);
    checkCudaError(err, "cudaMemcpyAsync failed for final h_repr");
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after final cudaMemcpyAsync");

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
    checkCudaError(cudaFree(input_repr), "cudaFree failed for final input_repr");
    checkCudaError(cudaFree(output_repr), "cudaFree failed for final output_repr");
    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy failed");

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
    checkCudaError(err, "cudaStreamCreate failed in solveGPU");

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
        checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy failed in solveGPU (early exit)");
        return;
    }

    uint32_t* h_repr = new uint32_t[19ULL * num_boards]();
    if (h_repr == nullptr) {
        fprintf(stderr, "new failed for h_repr\n");
        checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy failed in solveGPU (after new failure)");
        exit(1);
    }
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
    err = cudaMemcpyAsync(inputs_ptr->repr, h_repr, bytes, cudaMemcpyHostToDevice, stream);
    checkCudaError(err, "cudaMemcpyAsync failed for inputs_ptr");
    checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed after cudaMemcpyAsync for inputs_ptr");
    delete[] h_repr;

    std::vector<std::array<uint8_t, 81>> solutions = solve_multiple_sudoku(inputs_ptr);

    // End timing before writing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy failed in solveGPU");

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