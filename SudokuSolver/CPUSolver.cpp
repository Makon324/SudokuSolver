#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stack>

class board
{
    uint64_t repr[6] = {};

    void set_index(unsigned char index) {
        unsigned char array_index = index / 64;
        unsigned char bit_index = index % 64;
        repr[array_index] |= (1ULL << bit_index);
    }

    void unset_index(unsigned char index) {
        unsigned char array_index = index / 64;
        unsigned char bit_index = index % 64;
        repr[array_index] &= ~(1ULL << bit_index);
    }

    bool get_index(unsigned char index) {
        unsigned char array_index = index / 64;
        unsigned char bit_index = index % 64;
        return ((repr[array_index] & (1ULL << bit_index)) > 0);
	}

public:
    void set(unsigned char x, unsigned char y, unsigned char number) {
		set_index(9*x + number);
        set_index(9*9 + 9*y + number);
        set_index(2*9*9 + 9*((x/3) + 3*(y/3)) + number);
    }

    void unset(unsigned char x, unsigned char y, unsigned char number) {
        unset_index(9 * x + number);
        unset_index(9 * 9 + 9 * y + number);
        unset_index(2 * 9 * 9 + 9 * ((x / 3) + 3 * (y / 3)) + number);
    }

    bool is_blocked(unsigned char x, unsigned char y, unsigned char number) {
        return get_index(9*x + number) ||
               get_index(9*9 + 9*y + number) ||
               get_index(2*9*9 + 9*((x/3) + 3*(y/3)) + number);
	}

    bool is_preset(unsigned char x, unsigned char y) {
        unsigned char index = 3*9*9 + 9 * x + y;
        unsigned char array_index = index / 64;
        unsigned char bit_index = index % 64;
		return repr[array_index] & (1ULL << bit_index);
	}

    void set_preset(unsigned char x, unsigned char y) {
        unsigned char index = 3 * 9 * 9 + 9 * x + y;
        unsigned char array_index = index / 64;
        unsigned char bit_index = index % 64;
        repr[array_index] |= (1ULL << bit_index);
    }
};

board read_board_from_file(const std::string& filename) {
    board b;
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return b;  // Return empty board on failure
    }

    for (unsigned char x = 0; x < 9; x++) {  // x = row
        std::string line;
        if (!std::getline(in, line)) {
            std::cerr << "Incomplete file content." << std::endl;
            return b;
        }
        if (line.length() < 9) {
            std::cerr << "Line too short in file." << std::endl;
            continue;
        }
        for (unsigned char y = 0; y < 9; y++) {  // y = column
            char c = line[y];
            if (c >= '1' && c <= '9') {
                unsigned char num = c - '1';  // Convert '1'-'9' to 0-8
                if (!b.is_blocked(x, y, num)) {
                    b.set(x, y, num);
                    b.set_preset(x, y);
                }
                else {
                    std::cerr << "Invalid preset at (" << (int)x << ", " << (int)y << "): conflicts with existing constraints." << std::endl;
                }
            }
            // Ignore '0', '.', or other characters as empty cells
        }
    }
    return b;
}

bool solve(board& b, unsigned char start_x, unsigned char start_y) {
    std::vector<std::pair<unsigned char, unsigned char>> empties;
    bool started = false;
    for (unsigned char yy = 0; yy < 9; yy++) {
        for (unsigned char xx = 0; xx < 9; xx++) {
            if (!started) {
                if (yy > start_y || (yy == start_y && xx >= start_x)) {
                    started = true;
                }
                else {
                    continue;
                }
            }
            if (!b.is_preset(xx, yy)) {
                empties.push_back({ xx, yy });
            }
        }
    }
    if (empties.empty()) {
        return true;
    }
    std::stack<unsigned char> choice_stack;
    int ptr = 0;
    while (true) {
        auto [x, y] = empties[ptr];
        unsigned char start_k;
        bool is_new = (ptr == static_cast<int>(choice_stack.size()));
        if (is_new) {
            start_k = 0;
        }
        else {
            unsigned char prev_k = choice_stack.top();
            choice_stack.pop();
            b.unset(x, y, prev_k);
            start_k = prev_k + 1;
        }
        bool found = false;
        for (unsigned char k = start_k; k < 9; k++) {
            if (!b.is_blocked(x, y, k)) {
                b.set(x, y, k);
                choice_stack.push(k);
                found = true;
                ptr++;
                if (ptr == static_cast<int>(empties.size())) {
                    return true;
                }
                break;
            }
        }
        if (!found) {
            ptr--;
            if (ptr < 0) {
                return false;
            }
        }
    }
}


int main(int argc, ch)
{




    return 0;
}


