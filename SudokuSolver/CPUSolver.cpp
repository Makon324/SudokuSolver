#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>
#include <chrono>
#include <queue>

bool backtrack(std::array<uint8_t, 81>& board, int pos,
    std::array<uint16_t, 9>& rows,
    std::array<uint16_t, 9>& cols,
    std::array<uint16_t, 9>& boxes) {
    if (pos == 81) return true;
    if (board[pos] != 0) return backtrack(board, pos + 1, rows, cols, boxes);

    int r = pos / 9;
    int c = pos % 9;
    int b = (r / 3) * 3 + (c / 3);

    for (uint8_t num = 1; num <= 9; ++num) {
        uint16_t mask = 1 << (num - 1);
        if ((rows[r] & mask) == 0 && (cols[c] & mask) == 0 && (boxes[b] & mask) == 0) {
            board[pos] = num;
            rows[r] |= mask;
            cols[c] |= mask;
            boxes[b] |= mask;
            if (backtrack(board, pos + 1, rows, cols, boxes)) return true;
            board[pos] = 0;
            rows[r] &= ~mask;
            cols[c] &= ~mask;
            boxes[b] &= ~mask;
        }
    }
    return false;
}

bool solve_sudoku(std::array<uint8_t, 81>& board) {
    std::array<uint16_t, 9> rows = { 0 }, cols = { 0 }, boxes = { 0 };
    for (int pos = 0; pos < 81; ++pos) {
        uint8_t num = board[pos];
        if (num != 0) {
            uint16_t mask = 1 << (num - 1);
            int r = pos / 9;
            int c = pos % 9;
            int b = (r / 3) * 3 + (c / 3);
            if (rows[r] & mask || cols[c] & mask || boxes[b] & mask) return false;
            rows[r] |= mask;
            cols[c] |= mask;
            boxes[b] |= mask;
        }
    }
    return backtrack(board, 0, rows, cols, boxes);
}

std::vector<std::array<uint8_t, 81>> SolveSudokusCPU(std::vector<std::array<uint8_t, 81>>& input_boards) {
    std::vector<std::array<uint8_t, 81>> solutions;
    for (auto& input : input_boards) {
        auto board = input;
        if (solve_sudoku(board)) {
            solutions.push_back(board);
        }
        else {
            std::array<uint8_t, 81> unsolvable = { 0 };
            solutions.push_back(unsolvable);
        }
    }
    return solutions;
}

std::vector<std::array<uint8_t, 81>> read_boardsCPU(const std::string& filename, int count) {
    std::vector<std::array<uint8_t, 81>> boards;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return boards;
    }
    std::string line;
    int read = 0;
    while (std::getline(file, line) && read < count) {
        if (line.size() != 81) continue;
        std::array<uint8_t, 81> board;
        bool valid = true;
        for (size_t i = 0; i < 81; ++i) {
            char c = line[i];
            if (c < '0' || c > '9') {
                valid = false;
                break;
            }
            board[i] = c - '0';
        }
        if (valid) {
            boards.push_back(board);
            ++read;
        }
    }
    file.close();
    return boards;
}

void write_solutionsCPU(const std::string& filename, std::vector<std::array<uint8_t, 81>>& solutions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    for (auto& solution : solutions) {
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
    int count) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "Reading boards..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto boards = read_boardsCPU(input_file, count);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Read " << boards.size() << " boards in " << duration.count() << " seconds." << std::endl;

    std::cout << "Solving boards..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto solutions = SolveSudokusCPU(boards);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Solved " << solutions.size() << " boards in " << duration.count() << " seconds." << std::endl;

    std::cout << "Writing solutions..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    write_solutionsCPU(output_file, solutions);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Wrote solutions in " << duration.count() << " seconds." << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    duration = end_total - start_total;
    std::cout << "Total time: " << duration.count() << " seconds." << std::endl;
}