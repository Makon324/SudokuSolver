#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>

class SudokuBoard
{
    int id;

    uint32_t repr[11] = {}; // 4*9*9 = 324 bits -> 11 uint32_t (352 bits)

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
        return ((repr[array_index] & (1ULL << bit_index)) > 0);
    }

    inline void set_set(uint16_t index) {
        uint8_t array_index = index >> 5;  // index / 32
        uint8_t bit_index = index & 31;  // index % 32
        repr[array_index] |= (1U << bit_index);
    }

    inline void unset_set(uint16_t index) {
        uint8_t array_index = index >> 5;  // index / 32
        uint8_t bit_index = index & 31;  // index % 32
        repr[array_index] &= ~(1U << bit_index);
	}

    static const uint8_t row_id[81];
    static const uint8_t col_id[81];
    static const uint8_t box_id[81];

public:
    void set(uint8_t pos, uint8_t number) {
        set_index(9 * row_id[pos] + number);
        set_index(81 + 9 * col_id[pos] + number);
        set_index(162 + 9 * box_id[pos] + number);
		set_set(243 + pos);
    }

    bool is_set(uint16_t index) {
        uint8_t array_index = index >> 5;  // index / 32
        uint8_t bit_index = index & 31;  // index % 32
        return repr[array_index] & (1U << bit_index);
    }

    void unset(uint8_t pos, uint8_t number) {
        unset_index(9 * row_id[pos] + number);
        unset_index(81 + 9 * col_id[pos] + number);
        unset_index(162 + 9 * box_id[pos] + number);
		unset_set(243 + pos);
    }

    bool is_blocked(uint8_t pos, uint8_t number) {
        return get_index(9 * row_id[pos] + number) ||
            get_index(81 + 9 * col_id[pos] + number) ||
            get_index(162 + 9 * box_id[pos] + number);
            // is_set() called before
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

void generate_boards(std::vector<SudokuBoard>& boards) {
    std::vector<SudokuBoard> new_boards;
    for (auto &board : boards) {
        uint8_t pos = find_next_pos(board);
        if (pos == 255) {
			// impossible
            continue;
        }
        if (pos == 200) {
            // solved
            new_boards.push_back(board);
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




int main(int argc, char* argv[])
{




    return 0;
}


