#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <chrono>


class SudokuBoard
{
	uint32_t repr[18]{}; // 3*9*9 = 243 bits + 81 * 4 = 324 bits = 567 bits < 576 bits = 18 * 32 bits

    inline void set_index(uint16_t index) {
		uint8_t array_index = index >> 5;  // index / 32
		uint8_t bit_index = index & 31;  // index % 32
        repr[array_index] |= (1U << bit_index);
    }

    inline void unset_index(uint16_t index) {
        uint8_t array_index = index >> 5;  // index / 32
		uint8_t bit_index = index & 31;  // index % 32
        repr[array_index] &= ~(1U << bit_index);
    }

    inline bool get_index(uint16_t index) {
        uint8_t array_index = index >> 5;  // index / 32
		uint8_t bit_index = index & 31;  // index % 32
        return ((repr[array_index] & (1U << bit_index)) > 0);
    }

    inline void set_number_at_pos(uint8_t pos, uint8_t number) {
		uint8_t array_index = 17 - (pos >> 3);  // end - (pos / 8)
        uint8_t bit_index = 28 - ((pos & 7) << 2);  // 28 - (pos % 8) * 4
		repr[array_index] |= (number << bit_index);
    }

    inline void unset_number_at_pos(uint8_t pos) {
        uint8_t array_index = 17 - (pos >> 3);  // end - (pos / 8)
        uint8_t bit_index = 28 - ((pos & 7) << 2);  // 28 - (pos % 8) * 4
		repr[array_index] &= (-1U ^ (15 << bit_index));
    }

    static const uint8_t row_id[81];
    static const uint8_t col_id[81];
    static const uint8_t box_id[81];

public:
    int id;

    void set(uint8_t pos, uint8_t number) {
        set_index(9 * row_id[pos] + number);
        set_index(81 + 9 * col_id[pos] + number);
        set_index(162 + 9 * box_id[pos] + number);
		set_number_at_pos(pos, number + 1);
    }

    bool is_set(uint8_t pos) {
        uint8_t array_index = 17 - (pos >> 3);  // end - (pos / 8)
        uint8_t bit_index = 28 - ((pos & 7) << 2);  // 28 - (pos % 8) * 4
        return (repr[array_index] & (15 << bit_index)) > 0;
    }

    void unset(uint8_t pos, uint8_t number) {
        unset_index(9 * row_id[pos] + number);
        unset_index(81 + 9 * col_id[pos] + number);
        unset_index(162 + 9 * box_id[pos] + number);
		unset_number_at_pos(pos);
    }

    bool is_blocked(uint8_t pos, uint8_t number) {
        return get_index(9 * row_id[pos] + number) ||
            get_index(81 + 9 * col_id[pos] + number) ||
            get_index(162 + 9 * box_id[pos] + number);
            // is_set() called before
    }

    uint8_t get_number_at_pos(uint8_t pos) {
        uint8_t array_index = 17 - (pos >> 3);  // end - (pos / 8)
        uint8_t bit_index = 28 - ((pos & 7) << 2);  // 28 - (pos % 8) * 4
        return (repr[array_index] >> bit_index) & 15;
	}
};

const uint8_t SudokuBoard::row_id[81] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8
};

const uint8_t SudokuBoard::col_id[81] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8
};

const uint8_t SudokuBoard::box_id[81] = {
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    1, 1, 1, 4, 4, 4, 7, 7, 7,
    1, 1, 1, 4, 4, 4, 7, 7, 7,
    1, 1, 1, 4, 4, 4, 7, 7, 7,
    2, 2, 2, 5, 5, 5, 8, 8, 8,
    2, 2, 2, 5, 5, 5, 8, 8, 8,
    2, 2, 2, 5, 5, 5, 8, 8, 8
};

const int MAX_LEVELS = 81;

uint8_t find_next_pos(SudokuBoard& board) {
    int current_best = 10;
    uint8_t best_pos = -1;
    bool is_solved = true;
    for (uint8_t i = 0; i < 81; i++) {
        if (!board.is_set(i)) {
            is_solved = false;
            int num_possible = 0;
            for (uint8_t num = 0; num < 9; num++) {
                if (!board.is_blocked(i, num)) {
                    num_possible++;
                }
            }
            if (num_possible < current_best) {
                current_best = num_possible;
                best_pos = i;
            }
        }
    }
    if (is_solved) {
        return 200;
    }
    return current_best != 10 ? best_pos : 255;
}

// Small structure that represents one search state on the stack
struct SearchNode {
    SudokuBoard board;
    uint8_t pos;      // current position we are trying to fill (0..80), 255 = finished
};

// Solve a single puzzle with classic dancing-links-style backtracking (DFS)
// but implemented iteratively with std::stack ? pure BFS/DFS without recursion
bool solve_one_sudoku(SudokuBoard board, std::array<uint8_t, 81>& out_solution)
{
    struct StackFrame {
        uint8_t pos_to_fill;
        uint8_t candidate;   // next number to try at pos_to_fill (0..9)
    };

    std::stack<StackFrame> stk;
    uint8_t current_pos = find_next_pos(board);   // first empty cell with fewest possibilities

    if (current_pos == 200) {                     // already solved
        for (uint8_t i = 0; i < 81; ++i)
            out_solution[i] = board.get_number_at_pos(i);
        return true;
    }
    if (current_pos == 255)                       // impossible board
        return false;

    stk.push({ current_pos, 0 });

    while (!stk.empty()) {
        StackFrame& frame = stk.top();

        // Try the next candidate at the current position
        while (frame.candidate < 9) {
            uint8_t num = frame.candidate++;
            if (!board.is_blocked(frame.pos_to_fill, num)) {
                board.set(frame.pos_to_fill, num);
                uint8_t next_pos = find_next_pos(board);

                if (next_pos == 200) {                // solved!
                    for (uint8_t i = 0; i < 81; ++i)
                        out_solution[i] = board.get_number_at_pos(i);
                    return true;
                }
                if (next_pos != 255) {                // valid continuation
                    stk.push({ next_pos, 0 });
                    goto next_frame;                  // break out of candidate loop
                }
                // invalid branch ? undo and try next candidate
                board.unset(frame.pos_to_fill, num);
            }
        }

        // All candidates exhausted ? backtrack
        board.unset(frame.pos_to_fill, board.get_number_at_pos(frame.pos_to_fill) - 1);
        stk.pop();

    next_frame:;
    }

    return false;   // no solution found
}

// ---------------------------------------------------------------------------
// New, clean, one-by-one solver
// ---------------------------------------------------------------------------
std::vector<std::array<uint8_t, 81>> SolveSudokusCPU(const std::vector<SudokuBoard>& input_boards)
{
    const size_t n = input_boards.size();
    std::vector<std::array<uint8_t, 81>> solutions(n);

    for (size_t i = 0; i < n; ++i) {
        const SudokuBoard& puzzle = input_boards[i];
        std::array<uint8_t, 81>& result = solutions[i];

        // Fill with zeros = unsolved / empty
        result.fill(0);

        // Copy the board because solve_one_sudoku modifies it
        SudokuBoard board = puzzle;

        // Pre-fill already given clues into the result (they are guaranteed correct)
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t val = board.get_number_at_pos(pos);
            if (val != 0) {
                result[pos] = val;
            }
        }

        bool solved = solve_one_sudoku(board, result);

        if (!solved) {
            // Optional: you can leave it zero-filled or mark it specially
            // Here we keep zeros ? caller can detect unsolved puzzles easily
            std::cerr << "Puzzle " << i << " has no solution\n";
        }
    }

    return solutions;
}

std::vector<SudokuBoard> read_boardsCPU(const std::string& filename, const int count) {
    std::vector<SudokuBoard> boards;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return boards;
    }
    std::string line;
    int id = 0;
    while (std::getline(file, line) && id < count) {
        if (line.length() != 81) {
            continue; // Skip invalid lines
        }
        SudokuBoard board;
        bool valid = true;
        for (uint8_t pos = 0; pos < 81; ++pos) {
            char c = line[pos];
            if (c < '0' || c > '9') {
                valid = false;
                break;
            }
            if (c != '0') {
                uint8_t num = c - '0'; // 1-9
                uint8_t number = num - 1; // 0-8 for internal representation
                if (board.is_blocked(pos, number)) {
                    valid = false;
                    break; // Skip inconsistent puzzles
                }
                board.set(pos, number);
            }
        }
        if (valid) {
            board.id = id++;
            boards.push_back(board);
        }
    }
    file.close();
    return boards;
}

void write_solutionsCPU(const std::string& filename, const std::vector<std::array<uint8_t, 81>>& solutions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
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
            // Skip or handle unsolvable puzzles, e.g., write a placeholder
            file << "No solution" << std::endl;
            continue;
        }
        for (uint8_t pos = 0; pos < 81; ++pos) {
            uint8_t num = solution[pos];
            file << static_cast<char>('0' + num);
        }
        file << std::endl;
    }
    file.close();
}

void solveCPU(const std::string& input_file,
    const std::string& output_file,
    int count = std::numeric_limits<int>::max())
{
    std::vector<SudokuBoard> boards = read_boardsCPU(input_file, count);
    if (boards.empty()) {
        std::cerr << "No valid puzzles read.\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::array<uint8_t, 81>> solutions = SolveSudokusCPU(boards);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Solved " << boards.size()
        << " puzzles in " << elapsed.count() << " seconds\n";

    write_solutionsCPU(output_file, solutions);
}





