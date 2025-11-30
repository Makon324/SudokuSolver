#pragma once
#include <vector>

std::vector<SudokuBoard> read_boards(const std::string& filename);

std::vector<std::array<uint8_t, 81>> SolveSudokusCPU(std::vector<SudokuBoard>& boards);
