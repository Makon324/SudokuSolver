#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <stack>


void solveCPU(const std::string& input_file, const std::string& output_file, int count);

void solveGPU(const std::string& input_file, const std::string& output_file, int count);

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: sudoku method count input_file output_file" << std::endl;
        return 1;
    }

    std::string method = argv[1];
    int count;
    try {
        count = std::stoi(argv[2]);
    }
    catch (const std::exception&) {
        std::cout << "Invalid count: " << argv[2] << std::endl;
        return 1;
    }
    std::string input_file = argv[3];
    std::string output_file = argv[4];

    if (method == "cpu") {
        solveCPU(input_file, output_file, count);
    }
    else if (method == "gpu") {
        solveGPU(input_file, output_file, count);
    }
    else {
        std::cout << "Invalid method: " << method << ". Must be 'cpu' or 'gpu'." << std::endl;
        return 1;
    }

    return 0;
}