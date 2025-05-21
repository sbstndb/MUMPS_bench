#include "matrix_utils.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <fast_matrix_market/fast_matrix_market.hpp>

// Constructor implementation
template <typename INT, typename FLOAT>
c_matrix<INT, FLOAT>::c_matrix() : nrows(0), ncols(0) {
    // Initialize vectors if necessary, though their default constructors do this.
    // rows, cols, vals are default-initialized.
}

// read_matrix method implementation
template <typename INT, typename FLOAT>
c_matrix<INT, FLOAT>* c_matrix<INT, FLOAT>::read_matrix(std::string const &filename) {
    std::ifstream mfile;
    mfile.open(filename);
    if (!mfile.is_open()) {
        // It's good practice to handle file open errors.
        // For now, adhering to original, which didn't explicitly show error handling here.
        // Consider adding: throw std::runtime_error("Could not open matrix file: " + filename);
        // Or return nullptr and let caller check.
    }
    fast_matrix_market::read_options options;
    options.num_threads = 1; // As per original
    fast_matrix_market::read_matrix_market_triplet(mfile, nrows, ncols, rows, cols, vals, options);
    mfile.close(); // Good practice to close the file stream
    return this;
}

// Explicit template instantiations
// This is necessary because the template definitions are in a .cpp file.
template class c_matrix<int, double>;
// Add other instantiations if c_matrix is used with other types, e.g.:
// template class c_matrix<long, double>;
template class c_matrix<int, float>;
