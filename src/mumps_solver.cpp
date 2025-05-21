#include "mumps_solver.hpp"
#include <iostream> // For std::cout, std::cerr (used in some original methods, though not directly in c_mumps)
#include <vector>
#include <string>
#include <type_traits> // For std::is_same_v
#include <mpi.h>       // For MPI_Comm_rank, MPI_COMM_WORLD, etc.

// For MUMPS structures and functions
#include <dmumps_c.h>
#include <smumps_c.h>

// Constructor
#include <cstring> // For memset

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::c_mumps() : comm(USE_COMM_WORLD) {
    memset(&mumps, 0, sizeof(XMUMPS_STRUC_C));
    // cli, maps, mat members are default-constructed.
    // mumps struct is now zero-initialized.
}

// Methods
template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::apply_cli_params() {
    for (const auto &param : cli.icntl_params) {
        set_icntl(param.first, param.second);
    }
    for (const auto &param : cli.cntl_params) {
        set_cntl(param.first, param.second);
    }
    // cli.epsilon is double, cntl(7) expects FLOAT
    if (cli.epsilon != 0.0) { 
        set_cntl(7, static_cast<FLOAT>(cli.epsilon));
        set_icntl(35, 2); // Enable BLR
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::launch() {
    if constexpr (std::is_same_v<XMUMPS_STRUC_C, DMUMPS_STRUC_C>) {
        dmumps_c(&mumps);
    } else if constexpr (std::is_same_v<XMUMPS_STRUC_C, SMUMPS_STRUC_C>) {
        smumps_c(&mumps);
    } else {
        // This case should ideally not be reached if XMUMPS_STRUC_C is constrained.
        // static_assert(false, "Unsupported MUMPS structure type"); // Would fail compilation
        std::cerr << "Error: Unsupported MUMPS structure type in launch()" << std::endl;
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::launch(int job) {
    mumps.job = job;
    launch();
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::analysis() { launch(1); }

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::factorize() { launch(2); }

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::solve() { launch(3); }

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::compute_all() { launch(6); }

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::init_all_rank() {
    mumps.par = 1; // Host involved in computations
    mumps.sym = 0; // Unsymmetric matrix
    mumps.comm_fortran = comm; // Assign MPI_Comm directly to int (Fortran handle)
    launch(JOB_INIT);
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::init_master_rank() {
    int rank;
    MPI_Comm_rank(comm, &rank); // Use the communicator from the c_mumps object
    if (rank == 0) {
        // Adjust matrix indices from 0-based (fast_matrix_market) to 1-based (MUMPS)
        // This should be done only if matrix is freshly loaded and not yet adjusted.
        // Assuming mat.rows and mat.cols are 0-indexed here.
        for (size_t i = 0; i < mat.cols.size(); i++) {
            mat.cols[i] += 1;
            mat.rows[i] += 1;
        }
        mumps.n = mat.nrows;
        mumps.nnz = static_cast<MUMPS_INT>(mat.vals.size()); // MUMPS_INT is usually int
        
        // Correctly assign pointers for Fortran interface
        // For irn/jcn, MUMPS expects non-const pointers.
        // For a, MUMPS might expect non-const if it does in-place modifications (e.g. scaling).
        mumps.irn = mat.rows.data();
        mumps.jcn = mat.cols.data();
        mumps.a = mat.vals.data();
        
        set_rhs(); // Initialize rhs vector
        mumps.rhs = rhs.data();
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
int c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_cli(int argc, char **argv) {
    return cli.get_cli(argc, argv);
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::init() {
    init_all_rank();
    // Matrix must be loaded before init_master_rank if it relies on matrix dimensions/data
    // The original main.cpp calls set_matrix() before init().
    // If mat is not set, init_master_rank might operate on an empty matrix.
    init_master_rank();
}

///-------------- SETTERS
template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_matrix() {
    mat.read_matrix(cli.f_matrix); // mat is updated in-place
    // No need to call this->set_matrix(mat) as mat is already the member.
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_matrix(c_matrix<INT, FLOAT> const &matrix) {
    this->mat = matrix;
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_matrix_from_file(std::string const &filename) {
    this->mat.read_matrix(filename); // mat is updated in-place
}


template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_rhs() {
    if (mat.nrows > 0) {
        rhs.assign(mat.nrows, static_cast<FLOAT>(1.0));
    } else {
        // Handle case where matrix is not loaded or empty
        rhs.clear();
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_icntl(int key, INT value) {
    // MUMPS icntl/cntl arrays are 1-indexed in documentation, 0-indexed in C struct
    if (key > 0 && key <= sizeof(mumps.icntl) / sizeof(mumps.icntl[0])) {
        mumps.icntl[key - 1] = value;
    } else {
        std::cerr << "Warning: ICNTL key " << key << " is out of valid range." << std::endl;
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_cntl(int key, FLOAT value) {
    if (key > 0 && key <= sizeof(mumps.cntl) / sizeof(mumps.cntl[0])) {
        mumps.cntl[key - 1] = value;
    } else {
        std::cerr << "Warning: CNTL key " << key << " is out of valid range." << std::endl;
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_keep(int key, INT value) {
    // KEEP array is also 1-indexed in documentation
    if (key > 0 && key <= sizeof(mumps.keep) / sizeof(mumps.keep[0])) {
        mumps.keep[key - 1] = value;
    } else {
        std::cerr << "Warning: KEEP key " << key << " is out of valid range." << std::endl;
    }
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::set_blr(const FLOAT &epsilon) {
    set_icntl(35, 2); // Activate BLR based on element type
    set_cntl(7, epsilon); // Set BLR relaxation parameter
}

//---------------- GETTERS
template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
INT c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_info(int key) {
    if (key > 0 && key <= sizeof(mumps.info) / sizeof(mumps.info[0])) {
        return mumps.info[key - 1];
    }
    std::cerr << "Warning: INFO key " << key << " is out of valid range." << std::endl;
    return -1; // Or some other error indicator
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
INT c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_infog(int key) {
     if (key > 0 && key <= sizeof(mumps.infog) / sizeof(mumps.infog[0])) {
        return mumps.infog[key - 1];
    }
    std::cerr << "Warning: INFOG key " << key << " is out of valid range." << std::endl;
    return -1;
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
FLOAT c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_rinfo(int key) {
    if (key > 0 && key <= sizeof(mumps.rinfo) / sizeof(mumps.rinfo[0])) {
        return mumps.rinfo[key - 1];
    }
    std::cerr << "Warning: RINFO key " << key << " is out of valid range." << std::endl;
    return -1.0; // Or NAN
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
FLOAT c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_rinfog(int key) {
    if (key > 0 && key <= sizeof(mumps.rinfog) / sizeof(mumps.rinfog[0])) {
        return mumps.rinfog[key - 1];
    }
    std::cerr << "Warning: RINFOG key " << key << " is out of valid range." << std::endl;
    return -1.0;
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
c_mumps_information c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::get_all() {
    // Sizes from MUMPS documentation (can vary slightly by version, but these are common)
    // INFO/INFOG general size often up to 80. RINFO/RINFOG up to 40.
    // Using sizeof is safer if the struct definition is accurate.
    int max_info = sizeof(mumps.info) / sizeof(mumps.info[0]);
    int max_infog = sizeof(mumps.infog) / sizeof(mumps.infog[0]);
    int max_rinfo = sizeof(mumps.rinfo) / sizeof(mumps.rinfo[0]);
    int max_rinfog = sizeof(mumps.rinfog) / sizeof(mumps.rinfog[0]);

    c_mumps_information current_maps; // Create a temporary maps object

    for (int i = 0; i < max_info; i++) {
        current_maps.info[i + 1] = static_cast<long long int>(mumps.info[i]);
    }
    for (int i = 0; i < max_infog; i++) {
       current_maps.infog[i + 1] = static_cast<long long int>(mumps.infog[i]);
    }
    for (int i = 0; i < max_rinfo; i++) {
        current_maps.rinfo[i + 1] = static_cast<double>(mumps.rinfo[i]);
    }
    for (int i = 0; i < max_rinfog; i++) {
        current_maps.rinfog[i + 1] = static_cast<double>(mumps.rinfog[i]);
    }
    return current_maps;
}

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
void c_mumps<XMUMPS_STRUC_C, INT, FLOAT>::dump() {
    maps = get_all(); // Update the internal maps member
    maps.write_maps_to_file(cli.f_logs);
}

// Explicit template instantiations
template class c_mumps<DMUMPS_STRUC_C, int, double>;
template class c_mumps<SMUMPS_STRUC_C, int, float>; 
// Add other instantiations if c_mumps is used with other types.
// For example, if using long int for matrix indices or different float types.
// template class c_mumps<DMUMPS_STRUC_C, long int, double>;
// template class c_mumps<SMUMPS_STRUC_C, long int, float>;
