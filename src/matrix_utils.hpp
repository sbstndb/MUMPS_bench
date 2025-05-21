#ifndef C_MATRIX_HPP
#define C_MATRIX_HPP

#include <vector>
#include <string>
#include <fstream> // Required for std::ifstream in read_matrix implementation if moved here
// fast_matrix_market.hpp is needed for the implementation of read_matrix
#include <fast_matrix_market/fast_matrix_market.hpp>

template <typename INT, typename FLOAT>
class c_matrix {
public:
    INT nrows;
    INT ncols;
    std::vector<INT> rows, cols;
    std::vector<FLOAT> vals;

    c_matrix();

    // read_matrix reads data into the current instance and returns a pointer to it.
    c_matrix<INT, FLOAT>* read_matrix(std::string const &filename);
    // An alternative if we want to avoid raw pointers, could be to return by reference:
    // c_matrix<INT, FLOAT>& read_matrix(std::string const &filename);
    // Or make it void if the caller already has the object:
    // void read_matrix(std::string const &filename);
};

#endif // C_MATRIX_HPP
