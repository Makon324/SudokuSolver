#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <chrono>  // Added for timing


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

uint8_t find_next_pos(SudokuBoard &board){
    int current_best = 10;
	uint8_t best_pos;
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

void generate_boardsCPU(std::vector<SudokuBoard>& boards, int input_size) {
    std::vector<SudokuBoard> new_boards;
    std::vector<bool> is_solved(input_size, false);
    for (auto &board : boards) {
        if (is_solved[board.id]) {
            // already solved
            continue;
        }
        uint8_t pos = find_next_pos(board);
        if (pos == 200) {
            // solved
            is_solved[board.id] = true;
            new_boards.push_back(board);
            continue;
        }
        if (pos == 255) {
			// impossible
            continue;
        }
        for (uint8_t num = 0; num < 9; num++) {
            if (!board.is_blocked(pos, num)) {
                SudokuBoard new_board = board;
                new_board.set(pos, num);
                new_boards.push_back(new_board);
            }
        }
	}
	new_boards.shrink_to_fit();
	new_boards.swap(boards);
}

std::vector<std::array<uint8_t, 81>> SolveSudokusCPU(std::vector<SudokuBoard>& boards) {
	int input_size = boards.size();
    for (int level = 0; level < MAX_LEVELS; level++) {
		generate_boardsCPU(boards, input_size);
    }

	// Retrieve solutions
    std::vector<std::array<uint8_t, 81>> solutions(input_size);
	std::vector<bool> solution_found(input_size, false);
    for (auto &board : boards) {
		std::array<uint8_t, 81> solution{};
        if (solution_found[board.id]) {
            continue;
        }
        for (uint8_t pos = 0; pos < 81; pos++) {
			solution[pos] = board.get_number_at_pos(pos);
        }
		solutions[board.id] = solution;
        solution_found[board.id] = true;
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

void solveCPU(const std::string& input_file, const std::string& output_file, int count) {
    std::vector<SudokuBoard> boards = read_boardsCPU(input_file, count);
    if (boards.empty()) {
        return;
    }

    // Start timing after reading
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::array<uint8_t, 81>> solutions = SolveSudokusCPU(boards);

    // End timing before writing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    write_solutionsCPU(output_file, solutions);
}





