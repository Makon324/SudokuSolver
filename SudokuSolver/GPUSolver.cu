#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
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


// Constants for Sudoku dimensions and representation
constexpr int GRID_SIZE = 9;
constexpr int BOARD_SIZE = GRID_SIZE * GRID_SIZE; // 81
constexpr int SUBGRID_SIZE = 3;
constexpr int BITS_PER_GROUP = GRID_SIZE; // 9 bits for numbers 1-9
constexpr uint32_t MASK_GROUP = (1U << BITS_PER_GROUP) - 1; // 0x1FF
constexpr int ROW_MASK_BASE = 0;
constexpr int COL_MASK_BASE = BOARD_SIZE; // 81
constexpr int BOX_MASK_BASE = 2 * BOARD_SIZE; // 162
constexpr int BITS_PER_CELL = 4;
constexpr uint32_t MASK_CELL = (1U << BITS_PER_CELL) - 1; // 15
constexpr int CELLS_PER_FIELD = 32 / BITS_PER_CELL; // 8
constexpr int FIELDS_PER_GROUP = 6;
constexpr int ROW_FIELDS_START = 0;
constexpr int COL_FIELDS_START = ROW_FIELDS_START + FIELDS_PER_GROUP; // 6
constexpr int BOX_FIELDS_START = COL_FIELDS_START + FIELDS_PER_GROUP; // 12
constexpr int DATA_FIELDS_COUNT = FIELDS_PER_GROUP * 3; // 18
constexpr int LAST_DATA_FIELD = BOX_FIELDS_START + FIELDS_PER_GROUP - 1; // 17
constexpr int ID_FIELD = DATA_FIELDS_COUNT; // 18
constexpr int FIELDS_PER_BOARD = DATA_FIELDS_COUNT + 1; // 19
constexpr uint8_t SOLVED_CODE = 200;
constexpr uint8_t IMPOSSIBLE_CODE = 255;
constexpr int MAX_SOLVE_LEVELS = BOARD_SIZE;
constexpr int THREADS_PER_BLOCK = 256;
constexpr uint32_t MAX_PREALLOC_BOARDS = 1 << 20; // 1048576

/**
 * Class to manage representations of multiple Sudoku boards on GPU device memory.
 * Each board is compactly represented using FIELDS_PER_BOARD uint32_t values:
 * - Indices 0-5: Bitmasks for rows (BITS_PER_GROUP bits per row, but packed into uint32_t).
 * - Indices 6-11: Bitmasks for columns.
 * - Indices 12-17: Bitmasks for boxes.
 * - Index 18: Board ID for tracking original puzzles.
 */
class SudokuBoards
{
public:
    uint32_t num_boards;
    uint32_t* repr; // Device pointer to board representations

    /**
     * Constructor: Allocates device memory for the specified number of boards.
     */
    SudokuBoards(uint32_t n) : num_boards(n), repr(nullptr) {
        size_t bytes = static_cast<size_t>(FIELDS_PER_BOARD) * n * sizeof(uint32_t);
        cudaError_t err = cudaMalloc(&repr, bytes);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed in SudokuBoards constructor" << std::endl;
            exit(1);
        }
    }

    // Delete copy constructor and assignment to prevent unintended copies
    SudokuBoards(const SudokuBoards&) = delete;
    SudokuBoards& operator=(const SudokuBoards&) = delete;
    // Move constructor: Transfers ownership of device memory

    SudokuBoards(SudokuBoards&& other) noexcept : num_boards(other.num_boards), repr(other.repr) {
        other.repr = nullptr;
        other.num_boards = 0;
    }

    // Move assignment: Transfers ownership and frees existing memory if needed
    SudokuBoards& operator=(SudokuBoards&& other) noexcept {
        if (this != &other) {
            if (repr != nullptr) {
                cudaError_t err = cudaFree(repr);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed in SudokuBoards move assignment" << std::endl;
                    exit(1);
                }
            }
            num_boards = other.num_boards;
            repr = other.repr;
            other.num_boards = 0;
            other.repr = nullptr;
        }
        return *this;
    }

    /**
     * Destructor: Frees the device memory if allocated.
     */
    ~SudokuBoards() {
        if (repr != nullptr) {
            cudaError_t err = cudaFree(repr);
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed in SudokuBoards destructor" << std::endl;
                exit(1);
            }
        }
    }
    uint32_t get_num_boards() const { return num_boards; }
};

/**
 * Bit manipulation helper functions for Sudoku board representation.
 * These functions operate on bitmasks for rows, columns, boxes, and cell values.
 * They are marked __host__ __device__ to allow use on both CPU and GPU.
 */
 // Sets a specific bit in the bitmask representation
__host__ __device__ inline void set_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    repr[repr_idx * num_boards + board_idx] |= (1U << bit_index);
}

// Unsets a specific bit in the bitmask representation
__host__ __device__ inline void unset_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    repr[repr_idx * num_boards + board_idx] &= ~(1U << bit_index);
}

// Checks if a specific bit is set in the bitmask
__host__ __device__ inline bool get_index(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t index) {
    uint8_t repr_idx = index >> 5; // index / 32
    uint8_t bit_index = index & 31; // index % 32
    return ((repr[repr_idx * num_boards + board_idx] & (1U << bit_index)) > 0);
}

// Sets a number (1-9 stored as 1-9) at a specific position in the board
__host__ __device__ inline void set_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    uint8_t repr_idx = LAST_DATA_FIELD - (pos >> 3); // LAST_DATA_FIELD - (pos / 8)
    uint8_t bit_index = (32 - BITS_PER_CELL) - ((pos & (CELLS_PER_FIELD - 1)) << 2); // (32 - 4) - ((pos % 8) * 4)
    repr[repr_idx * num_boards + board_idx] |= (number << bit_index);
}

// Unsets the number at a specific position
__host__ __device__ inline void unset_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = LAST_DATA_FIELD - (pos >> 3); // LAST_DATA_FIELD - (pos / 8)
    uint8_t bit_index = (32 - BITS_PER_CELL) - ((pos & (CELLS_PER_FIELD - 1)) << 2); // (32 - 4) - ((pos % 8) * 4)
    repr[repr_idx * num_boards + board_idx] &= (-1U ^ (MASK_CELL << bit_index));
}

// Checks if a position has a set number
__host__ __device__ inline bool is_set(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = LAST_DATA_FIELD - (pos >> 3); // LAST_DATA_FIELD - (pos / 8)
    uint8_t bit_index = (32 - BITS_PER_CELL) - ((pos & (CELLS_PER_FIELD - 1)) << 2); // (32 - 4) - ((pos % 8) * 4)
    return (repr[repr_idx * num_boards + board_idx] & (MASK_CELL << bit_index)) > 0;
}

// Retrieves the number (1-9) at a specific position
__host__ __device__ inline uint8_t get_number_at_pos(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint8_t repr_idx = LAST_DATA_FIELD - (pos >> 3); // LAST_DATA_FIELD - (pos / 8)
    uint8_t bit_index = (32 - BITS_PER_CELL) - ((pos & (CELLS_PER_FIELD - 1)) << 2); // (32 - 4) - ((pos % 8) * 4)
    return (repr[repr_idx * num_boards + board_idx] >> bit_index) & MASK_CELL;
}

// Checks if placing a number at a position is blocked by row, column, or box constraints
__host__ __device__ inline bool is_blocked(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    return get_index(repr, num_boards, board_idx, ROW_MASK_BASE + GRID_SIZE * (pos / GRID_SIZE) + number) ||
        get_index(repr, num_boards, board_idx, COL_MASK_BASE + GRID_SIZE * (pos % GRID_SIZE) + number) ||
        get_index(repr, num_boards, board_idx, BOX_MASK_BASE + GRID_SIZE * (((pos / GRID_SIZE) / SUBGRID_SIZE) * SUBGRID_SIZE + ((pos % GRID_SIZE) / SUBGRID_SIZE)) + number);
}

// Sets a number at a position and updates the row, column, and box bitmasks
__host__ __device__ inline void set(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    set_index(repr, num_boards, board_idx, ROW_MASK_BASE + GRID_SIZE * (pos / GRID_SIZE) + number);
    set_index(repr, num_boards, board_idx, COL_MASK_BASE + GRID_SIZE * (pos % GRID_SIZE) + number);
    set_index(repr, num_boards, board_idx, BOX_MASK_BASE + GRID_SIZE * (((pos / GRID_SIZE) / SUBGRID_SIZE) * SUBGRID_SIZE + ((pos % GRID_SIZE) / SUBGRID_SIZE)) + number);
    set_number_at_pos(repr, num_boards, board_idx, pos, number + 1);
}

// Unsets a number at a position and clears the corresponding bitmasks
__host__ __device__ inline void unset(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos, uint8_t number) {
    unset_index(repr, num_boards, board_idx, ROW_MASK_BASE + GRID_SIZE * (pos / GRID_SIZE) + number);
    unset_index(repr, num_boards, board_idx, COL_MASK_BASE + GRID_SIZE * (pos % GRID_SIZE) + number);
    unset_index(repr, num_boards, board_idx, BOX_MASK_BASE + GRID_SIZE * (((pos / GRID_SIZE) / SUBGRID_SIZE) * SUBGRID_SIZE + ((pos % GRID_SIZE) / SUBGRID_SIZE)) + number);
    unset_number_at_pos(repr, num_boards, board_idx, pos);
}

// Retrieves the original board ID stored in the representation
__host__ __device__ inline uint32_t get_id(uint32_t* repr, uint32_t num_boards, uint32_t board_idx) {
    return repr[ID_FIELD * num_boards + board_idx];
}

/**
 * Retrieves a 9-bit mask (BITS_PER_GROUP bits) representing the used numbers (1-9) in a specific Sudoku group
 * (row, column, or 3x3 box) from the compact bit-packed representation of multiple Sudoku boards.
 *
 * The mask is extracted starting from the specified base_index in the logical bit array formed by the
 * board's mask fields in 'repr'. Each bit in the returned mask corresponds to a number: bit 0 for 1,
 * bit 1 for 2, ..., bit 8 for 9. If a bit is set (1), the number is used in the group.
 *
 * Handles cases where the 9 bits straddle two adjacent uint32_t fields by fetching and combining bits from both.
 */
__device__ inline uint32_t get_mask(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint16_t base_index) {
    uint8_t repr_idx = base_index >> 5; // base_index / 32
    uint8_t bit_index = base_index & 31; // base_index % 32
    uint32_t val1 = repr[repr_idx * num_boards + board_idx];
    if (bit_index <= (32 - BITS_PER_GROUP)) {
        return (val1 >> bit_index) & MASK_GROUP;
    }
    else {
        uint32_t val2 = repr[(repr_idx + 1) * num_boards + board_idx];
        uint32_t low = val1 >> bit_index;
        uint32_t high = val2 << (32 - bit_index);
        return (low | high) & MASK_GROUP;
    }
}

/**
 * Kernel to find the next cell for each board using MRV heuristic.
 * Performs naked and hidden singles propagation until no more changes or impossibility detected.
 * Outputs next position (or special codes: IMPOSSIBLE_CODE for impossible, SOLVED_CODE for solved)
 * and number of children (possible values for the MRV cell).
 */
__global__ void find_next_cell_kernel(uint32_t* d_repr, uint32_t num_boards, uint8_t* d_next_pos, uint32_t* d_num_children_out) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= num_boards) return;

    bool changed = true;
    bool impossible = false;
    int min_poss = GRID_SIZE + 1;
    uint8_t best_pos = 0;
    bool is_solved = false;

    while (changed && !impossible) {
        changed = propagate_naked_singles(d_repr, num_boards, board_idx, min_poss, best_pos, is_solved, impossible);

        if (impossible) break;

        if (propagate_hidden_singles(d_repr, num_boards, board_idx)) {
            changed = true;
        }
    }

    if (impossible) {
        d_next_pos[board_idx] = IMPOSSIBLE_CODE;
        d_num_children_out[board_idx] = 0;
        return;
    }

    uint32_t num_children;
    if (is_solved) {
        d_next_pos[board_idx] = SOLVED_CODE;
        num_children = 1;
    }
    else {
        d_next_pos[board_idx] = best_pos;
        num_children = min_poss;
    }
    d_num_children_out[board_idx] = num_children;
}

/**
 * Device function to compute the mask of available numbers for a position.
 * Combines masks from row, column, and box to find unused numbers (1-9).
 */
__device__ inline uint32_t get_available_mask(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint16_t base_row = ROW_MASK_BASE + GRID_SIZE * (pos / GRID_SIZE);
    uint16_t base_col = COL_MASK_BASE + GRID_SIZE * (pos % GRID_SIZE);
    uint16_t base_box = BOX_MASK_BASE + GRID_SIZE * (((pos / GRID_SIZE) / SUBGRID_SIZE) * SUBGRID_SIZE + ((pos % GRID_SIZE) / SUBGRID_SIZE));
    uint32_t row_m = get_mask(repr, num_boards, board_idx, base_row);
    uint32_t col_m = get_mask(repr, num_boards, board_idx, base_col);
    uint32_t box_m = get_mask(repr, num_boards, board_idx, base_box);
    uint32_t used = row_m | col_m | box_m;
    return ~used & MASK_GROUP;
}

/**
 * Kernel to generate child boards for each input board.
 * For solved boards, copies as is. For unsolved, creates one child per possible number at the next position.
 * Uses prefixes to determine output positions.
 */
__global__ void generate_children_kernel(uint32_t* d_in_repr, uint32_t in_num_boards, uint8_t* d_next_pos, uint32_t* d_prefixes, uint32_t* d_out_repr, uint32_t out_num_boards) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= in_num_boards) return; // if outside of id range

    uint8_t pos = d_next_pos[board_idx];
    if (pos == IMPOSSIBLE_CODE) return; // impossible, no children

    uint32_t out_start = d_prefixes[board_idx];
    if (pos == SOLVED_CODE) {
        // solved, copy as is
        uint32_t out_idx = out_start;
        for (uint8_t field = 0; field < FIELDS_PER_BOARD; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        return;
    }

    // Generate children for unsolved board
    uint32_t avail = get_available_mask(d_in_repr, in_num_boards, board_idx, pos);
    uint32_t child_idx = 0;
    while (avail != 0) {
        uint32_t bit = __ffs(avail) - 1;
        uint32_t out_idx = out_start + child_idx;
        for (uint8_t field = 0; field < FIELDS_PER_BOARD; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        set(d_out_repr, out_num_boards, out_idx, pos, bit);
        avail &= ~(1u << bit);
        ++child_idx;
    }
}

/**
 * Device function to compute the mask of available numbers for a position.
 * Combines masks from row, column, and box to find unused numbers (1-9).
 */
__device__ inline uint32_t get_available_mask(uint32_t* repr, uint32_t num_boards, uint32_t board_idx, uint8_t pos) {
    uint16_t base_row = ROW_MASK_BASE + GRID_SIZE * (pos / GRID_SIZE);
    uint16_t base_col = COL_MASK_BASE + GRID_SIZE * (pos % GRID_SIZE);
    uint16_t base_box = BOX_MASK_BASE + GRID_SIZE * (((pos / GRID_SIZE) / SUBGRID_SIZE) * SUBGRID_SIZE + ((pos % GRID_SIZE) / SUBGRID_SIZE));
    uint32_t row_m = get_mask(repr, num_boards, board_idx, base_row);
    uint32_t col_m = get_mask(repr, num_boards, board_idx, base_col);
    uint32_t box_m = get_mask(repr, num_boards, board_idx, base_box);
    uint32_t used = row_m | col_m | box_m;
    return ~used & MASK_GROUP;
}

/**
 * Kernel to generate child boards for each input board.
 * For solved boards, copies as is. For unsolved, creates one child per possible number at the next position.
 * Uses prefixes to determine output positions.
 */
__global__ void generate_children_kernel(uint32_t* d_in_repr, uint32_t in_num_boards, uint8_t* d_next_pos, uint32_t* d_prefixes, uint32_t* d_out_repr, uint32_t out_num_boards) {
    uint32_t board_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (board_idx >= in_num_boards) return; // if outside of id range

    uint8_t pos = d_next_pos[board_idx];
    if (pos == IMPOSSIBLE_CODE) return; // impossible, no children

    uint32_t out_start = d_prefixes[board_idx];
    if (pos == SOLVED_CODE) {
        // solved, copy as is
        uint32_t out_idx = out_start;
        for (uint8_t field = 0; field < FIELDS_PER_BOARD; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        return;
    }

    // Generate children for unsolved board
    uint32_t avail = get_available_mask(d_in_repr, in_num_boards, board_idx, pos);
    uint32_t child_idx = 0;
    while (avail != 0) {
        uint32_t bit = __ffs(avail) - 1;
        uint32_t out_idx = out_start + child_idx;
        for (uint8_t field = 0; field < FIELDS_PER_BOARD; ++field) {
            d_out_repr[field * out_num_boards + out_idx] = d_in_repr[field * in_num_boards + board_idx];
        }
        set(d_out_repr, out_num_boards, out_idx, pos, bit);
        avail &= ~(1u << bit);
        ++child_idx;
    }
}

/**
 * Computes the total number of new boards (sum of children) using Thrust reduce.
 */
uint32_t compute_new_num_boards(uint32_t* d_num_children_out, uint32_t num_boards, cudaStream_t stream) {
    return thrust::reduce(thrust::cuda::par.on(stream),
        thrust::device_ptr<uint32_t>(d_num_children_out),
        thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
        0u);
}

/**
 * Counts the number of unsolved boards using Thrust count_if.
 */
uint32_t count_unsolved(uint8_t* d_next_pos, uint32_t num_boards, cudaStream_t stream) {
    return thrust::count_if(thrust::cuda::par.on(stream),
        thrust::device_ptr<uint8_t>(d_next_pos),
        thrust::device_ptr<uint8_t>(d_next_pos + num_boards),
        [] __device__(uint8_t pos) { return pos < SOLVED_CODE; });
}

/**
 * Computes exclusive scan prefixes for child offsets using Thrust.
 */
void compute_prefixes(uint32_t* d_num_children_out, uint32_t* d_prefixes, uint32_t num_boards, cudaStream_t stream) {
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
        thrust::device_ptr<uint32_t>(d_num_children_out),
        thrust::device_ptr<uint32_t>(d_num_children_out + num_boards),
        thrust::device_ptr<uint32_t>(d_prefixes));
}

/**
 * Ensures the output buffer is large enough; reallocates if necessary.
 */
void ensure_buffer_size(uint32_t** output_repr, uint32_t* output_max, uint32_t new_num) {
    if (new_num > *output_max) {
        cudaError_t err = cudaFree(*output_repr);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for output_repr (overflow)" << std::endl;
            exit(1);
        }
        size_t new_bytes = static_cast<size_t>(FIELDS_PER_BOARD) * new_num * sizeof(uint32_t);
        err = cudaMalloc(output_repr, new_bytes);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for output_repr (overflow)" << std::endl;
            exit(1);
        }
        *output_max = new_num;
    }
}

/**
 * Extracts solutions from the final host representations.
 * Maps back to original puzzle IDs and constructs BOARD_SIZE-element arrays with numbers 1-9.
 */
std::vector<std::array<uint8_t, BOARD_SIZE>> extract_solutions(uint32_t* h_repr, uint32_t final_num_boards, uint32_t original_num) {
    std::vector<std::array<uint8_t, BOARD_SIZE>> solutions(original_num);
    std::vector<bool> found(original_num, false);
    for (uint32_t board_idx = 0; board_idx < final_num_boards; ++board_idx) {
        uint32_t id = get_id(h_repr, final_num_boards, board_idx);
        if (id >= original_num || found[id]) continue;
        auto& sol = solutions[id];
        for (uint8_t pos = 0; pos < BOARD_SIZE; ++pos) {
            uint8_t num = get_number_at_pos(h_repr, final_num_boards, board_idx, pos) - 1;
            sol[pos] = num + 1;
        }
        found[id] = true;
    }
    return solutions;
}

/**
 * Performs a single iteration of the solving loop: finds next cells, propagates singles,
 * generates children for unsolved boards, filters impossibles, and swaps buffers.
 * Returns true if all boards are solved (or pool empty), false to continue.
 */
bool single_loop_iteration(uint32_t*& input_repr, uint32_t*& output_repr, uint32_t& output_max, uint32_t& input_max, cudaStream_t& stream, uint32_t& num_boards) {
    std::cout << "Loop, Boards: " << num_boards << std::endl;
    if (num_boards == 0) return true;

    // Allocate temporary device arrays for next positions and child counts
    uint8_t* d_next_pos = nullptr;
    cudaError_t err = cudaMalloc(&d_next_pos, num_boards * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for d_next_pos" << std::endl;
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    uint32_t* d_num_children_out = nullptr;
    err = cudaMalloc(&d_num_children_out, num_boards * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for d_num_children_out" << std::endl;
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Launch kernel to find next cells, fill 1's and hidden singels
    int threads = THREADS_PER_BLOCK;
    int blocks = (num_boards + threads - 1) / threads;
    find_next_cell_kernel << <blocks, threads, 0, stream >> > (input_repr, num_boards, d_next_pos, d_num_children_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - find_next_cell_kernel launch failed" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after find_next_cell_kernel" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Compute total new boards
    uint32_t new_num = compute_new_num_boards(d_num_children_out, num_boards, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after thrust::reduce" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    if (new_num == 0) {
        // If no new no need to launch kernel
        num_boards = 0;
        err = cudaFree(d_next_pos);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_next_pos (no boards)" << std::endl;
            cudaFree(d_num_children_out);
            cudaFree(input_repr);
            cudaFree(output_repr);
            cudaStreamDestroy(stream);
            exit(1);
        }

        err = cudaFree(d_num_children_out);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_num_children_out (no boards)" << std::endl;
            cudaFree(input_repr);
            cudaFree(output_repr);
            cudaStreamDestroy(stream);
            exit(1);
        }

        return true;
    }

    // Count unsolved boards
    uint32_t unsolved_count = count_unsolved(d_next_pos, num_boards, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after thrust::count_if" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    bool all_solved = (unsolved_count == 0);
    if (all_solved && new_num == num_boards) {
        // All boards solved, no impossibles to filter, no need for generate kernel
        err = cudaFree(d_next_pos);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_next_pos (early exit)" << std::endl;
            cudaFree(d_num_children_out);
            cudaFree(input_repr);
            cudaFree(output_repr);
            cudaStreamDestroy(stream);
            exit(1);
        }

        err = cudaFree(d_num_children_out);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_num_children_out (early exit)" << std::endl;
            cudaFree(input_repr);
            cudaFree(output_repr);
            cudaStreamDestroy(stream);
            exit(1);
        }

        return true;
    }

    // Compute prefix sums to make generate_children_kernel know where to put what
    uint32_t* d_prefixes = nullptr;
    err = cudaMalloc(&d_prefixes, num_boards * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for d_prefixes" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    compute_prefixes(d_num_children_out, d_prefixes, num_boards, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after thrust::exclusive_scan" << std::endl;
        cudaFree(d_prefixes);
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Ensure output buffer is large enough
    ensure_buffer_size(&output_repr, &output_max, new_num);

    // Launch kernel to generate children
    generate_children_kernel << <blocks, threads, 0, stream >> > (input_repr, num_boards, d_next_pos, d_prefixes, output_repr, new_num);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - generate_children_kernel launch failed" << std::endl;
        cudaFree(d_prefixes);
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after generate_children_kernel" << std::endl;
        cudaFree(d_prefixes);
        cudaFree(d_num_children_out);
        cudaFree(d_next_pos);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Free temporary arrays
    err = cudaFree(d_next_pos);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_next_pos" << std::endl;
        cudaFree(d_num_children_out);
        cudaFree(d_prefixes);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaFree(d_num_children_out);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_num_children_out" << std::endl;
        cudaFree(d_prefixes);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaFree(d_prefixes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for d_prefixes" << std::endl;
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Swap input and output buffers for next level
    std::swap(input_repr, output_repr);
    std::swap(input_max, output_max);
    
    num_boards = new_num;  // Update num_boards after generation
    return all_solved;
}

/**
 * Solves multiple Sudoku puzzles using GPU-accelerated backtracking.
 * Uses two buffers for input/output swapping across levels.
 * Returns the solutions as vectors of BOARD_SIZE-element arrays.
 */
std::vector<std::array<uint8_t, BOARD_SIZE>> solve_multiple_sudoku(SudokuBoards* current) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamCreate failed in solve_multiple_sudoku" << std::endl;
        delete current;
        exit(1);
    }

    uint32_t original_num = current->get_num_boards();
    size_t prealloc_bytes = static_cast<size_t>(FIELDS_PER_BOARD) * MAX_PREALLOC_BOARDS * sizeof(uint32_t);
    uint32_t* repr_a = nullptr;  // 'a' SudokuBoards represenation
    err = cudaMalloc(&repr_a, prealloc_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for repr_a" << std::endl;
        cudaStreamDestroy(stream);
        delete current;
        exit(1);
    }

    uint32_t* repr_b = nullptr;  // 'b' SudokuBoards representation
    err = cudaMalloc(&repr_b, prealloc_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMalloc failed for repr_b" << std::endl;
        cudaFree(repr_a);
        cudaStreamDestroy(stream);
        delete current;
        exit(1);
    }

    size_t initial_bytes = static_cast<size_t>(FIELDS_PER_BOARD) * original_num * sizeof(uint32_t);
    err = cudaMemcpy(repr_a, current->repr, initial_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMemcpy failed for initial copy" << std::endl;
        cudaFree(repr_a);
        cudaFree(repr_b);
        cudaStreamDestroy(stream);
        delete current;
        exit(1);
    }

    delete current;
    uint32_t* input_repr = repr_a;
    uint32_t input_max = MAX_PREALLOC_BOARDS;
    uint32_t* output_repr = repr_b;
    uint32_t output_max = MAX_PREALLOC_BOARDS;
    uint32_t num_boards = original_num;

    // Process until all solved or no boards left
    for (int loop = 0; loop < MAX_SOLVE_LEVELS; ++loop) {
        bool done = single_loop_iteration(input_repr, output_repr, output_max, input_max, stream, num_boards);
        if (done) {
            break;
        }
    }

    // Copy from device
    uint32_t final_num_boards = num_boards;
    size_t bytes = static_cast<size_t>(FIELDS_PER_BOARD) * final_num_boards * sizeof(uint32_t);
    uint32_t* h_repr = (uint32_t*)malloc(bytes);
    if (h_repr == nullptr) {
        std::cerr << "malloc failed for h_repr" << std::endl;
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaMemcpyAsync(h_repr, input_repr, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMemcpyAsync failed for final h_repr" << std::endl;
        free(h_repr);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after final cudaMemcpyAsync" << std::endl;
        free(h_repr);
        cudaFree(input_repr);
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }

    // Extract solutions from host data
    auto solutions = extract_solutions(h_repr, final_num_boards, original_num);
    free(h_repr);

    // Clean up final buffers and stream
    err = cudaFree(input_repr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for final input_repr" << std::endl;
        cudaFree(output_repr);
        cudaStreamDestroy(stream);
        exit(1);
    }
    err = cudaFree(output_repr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaFree failed for final output_repr" << std::endl;
        cudaStreamDestroy(stream);
        exit(1);
    }
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamDestroy failed" << std::endl;
        exit(1);
    }
    return solutions;
}

/**
 * Reads Sudoku puzzles from the input file, validates them, and prepares representations.
 * Reads up to 'count' valid puzzles.
 */
std::vector<std::vector<uint32_t>> read_puzzles(const std::string& input_file, int count) {
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_file << std::endl;
        return {};
    }
    std::string line;
    std::vector<std::vector<uint32_t>> temp_reprs;
    int id = 0;
    while (std::getline(file, line) && id < count) {
        if (line.length() != BOARD_SIZE) continue;
        std::vector<uint32_t> h_single_repr(FIELDS_PER_BOARD, 0);
        bool valid = true;
        for (uint8_t pos = 0; pos < BOARD_SIZE; ++pos) {
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
    return temp_reprs;
}

/**
 * Prepares the host array for board representations, including IDs.
 */
uint32_t* prepare_host_repr(const std::vector<std::vector<uint32_t>>& temp_reprs) {
    uint32_t num_boards = temp_reprs.size();
    uint32_t* h_repr = new uint32_t[static_cast<size_t>(FIELDS_PER_BOARD) * num_boards]();
    if (h_repr == nullptr) {
        std::cerr << "new failed for h_repr" << std::endl;
        exit(1);
    }

    for (uint32_t b = 0; b < num_boards; ++b) {
        for (uint8_t f = 0; f < DATA_FIELDS_COUNT; ++f) {
            h_repr[f * num_boards + b] = temp_reprs[b][f];
        }
        h_repr[ID_FIELD * num_boards + b] = b;
    }

    return h_repr;
}

/**
 * Writes the solved puzzles to the output file.
 * Outputs "No solution" for unsolved puzzles.
 */
void write_solutions(const std::string& output_file, const std::vector<std::array<uint8_t, BOARD_SIZE>>& solutions) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    for (const auto& solution : solutions) {
        bool is_valid_solution = true;
        for (uint8_t pos = 0; pos < BOARD_SIZE; ++pos) {
            if (solution[pos] == 0) {
                is_valid_solution = false;
                break;
            }
        }
        if (!is_valid_solution) {
            out << "No solution" << std::endl;
            continue;
        }
        for (uint8_t pos = 0; pos < BOARD_SIZE; ++pos) {
            uint8_t num = solution[pos];
            out << static_cast<char>('0' + num);
        }
        out << std::endl;
    }

    out.close();
}

/**
 * Main entry point to solve Sudoku puzzles from file using GPU.
 * Reads input, prepares data, solves, times the solving process, and writes output.
 */
void solveGPU(const std::string& input_file, const std::string& output_file, int count) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamCreate failed in solveGPU" << std::endl;
        exit(1);
    }

    // Read and validate puzzles
    auto temp_reprs = read_puzzles(input_file, count);
    if (temp_reprs.empty()) {
        err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamDestroy failed in solveGPU (early exit)" << std::endl;
            exit(1);
        }
        return;
    }
    uint32_t num_boards = temp_reprs.size();

    // Prepare host representation array
    uint32_t* h_repr = prepare_host_repr(temp_reprs);
    temp_reprs.clear();

    // Start timing after reading and preparation
    auto start = std::chrono::high_resolution_clock::now();

    // Copy to device and solve
    SudokuBoards* inputs_ptr = new SudokuBoards(num_boards);
    size_t bytes = static_cast<size_t>(FIELDS_PER_BOARD) * num_boards * sizeof(uint32_t);
    err = cudaMemcpyAsync(inputs_ptr->repr, h_repr, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaMemcpyAsync failed for inputs_ptr" << std::endl;
        cudaStreamDestroy(stream);
        delete inputs_ptr;
        exit(1);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamSynchronize failed after cudaMemcpyAsync for inputs_ptr" << std::endl;
        cudaStreamDestroy(stream);
        delete inputs_ptr;
        exit(1);
    }
    delete[] h_repr;
    std::vector<std::array<uint8_t, BOARD_SIZE>> solutions = solve_multiple_sudoku(inputs_ptr);

    // End timing before writing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - cudaStreamDestroy failed in solveGPU" << std::endl;
        exit(1);
    }

    // Write solutions to file
    write_solutions(output_file, solutions);
}