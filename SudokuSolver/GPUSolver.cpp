#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>



class SudokuBoard
{
    int id;

	uint32_t repr[11] = {}; // 4*9*9 = 324 bits -> 11 uint32_t (352 bits)

    void set_index(unsigned char index) {
        unsigned char array_index = index / 32;
        unsigned char bit_index = index % 32;
        repr[array_index] |= (1ULL << bit_index);
    }

    void unset_index(unsigned char index) {
        unsigned char array_index = index / 32;
        unsigned char bit_index = index % 32;
        repr[array_index] &= ~(1ULL << bit_index);
    }

    bool get_index(unsigned char index) {
        unsigned char array_index = index / 32;
        unsigned char bit_index = index % 32;
        return ((repr[array_index] & (1ULL << bit_index)) > 0);
    }

public:
    void set(unsigned char x, unsigned char y, unsigned char number) {
        set_index(9 * x + number);
        set_index(9 * 9 + 9 * y + number);
        set_index(2 * 9 * 9 + 9 * ((x / 3) + 3 * (y / 3)) + number);
    }

    void unset(unsigned char x, unsigned char y, unsigned char number) {
        unset_index(9 * x + number);
        unset_index(9 * 9 + 9 * y + number);
        unset_index(2 * 9 * 9 + 9 * ((x / 3) + 3 * (y / 3)) + number);
    }

    bool is_blocked(unsigned char x, unsigned char y, unsigned char number) {
        return get_index(9 * x + number) ||
            get_index(9 * 9 + 9 * y + number) ||
            get_index(2 * 9 * 9 + 9 * ((x / 3) + 3 * (y / 3)) + number);
    }

    bool is_preset(unsigned char x, unsigned char y) {
        unsigned char index = 3 * 9 * 9 + 9 * x + y;
        unsigned char array_index = index / 32;
        unsigned char bit_index = index % 32;
        return repr[array_index] & (1ULL << bit_index);
    }

    void set_preset(unsigned char x, unsigned char y) {
        unsigned char index = 3 * 9 * 9 + 9 * x + y;
        unsigned char array_index = index / 32;
        unsigned char bit_index = index % 32;
        repr[array_index] |= (1ULL << bit_index);
    }
};


int MAX_LEVELS = 81;

// Kernel 1: Find Next Cell (MRV), track solved and invalid boards (solved x, y = 200; invalid x, y = 255)
__global__ void find_next_cell_kernel(SudokuBoard* boards, uint8_t* next_cells_x, uint8_t* next_cells_y) {

}

// Kernel 2: Generate Children
__global__ void generate_children_kernel(SudokuBoard* in_boards, uint8_t* next_cells_x, uint8_t* next_cells_y, SudokuBoard* out_boards) {

}

// Main function
void solve_multiple_sudoku(const std::vector<SudokuBoard>& inputs) {
	// Allocate device memory

    for (int level = 0; level < MAX_LEVELS; level++) {
        
		// launch find_next_cell_kernel

        // allocate memory, output finished bords

		// launch generate_children_kernel
        
    }
}





