#ifndef C_MUMPS_SOLVER_HPP
#define C_MUMPS_SOLVER_HPP

#include <mpi.h>
#include <vector>
#include <string>
#include <type_traits> // For std::is_same_v
#include <dmumps_c.h>
#include <smumps_c.h>

#include "cli.hpp"
#include "mumps_log.hpp"
#include "matrix_utils.hpp" // This provides c_matrix

// Define constants as specified
constexpr int USE_COMM_WORLD = -987654;
constexpr int JOB_INIT = -1;
constexpr int JOB_END = -2;

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
class c_mumps {
public:
    XMUMPS_STRUC_C mumps;
    MPI_Comm comm;
    c_matrix<INT, FLOAT> mat;
    std::vector<FLOAT> rhs;
    c_mumps_information maps;
    c_cli cli;

    // Constructor
    c_mumps();

    // Methods
    void apply_cli_params();
    void launch();
    void launch(int job);
    void analysis();
    void factorize();
    void solve();
    void compute_all();
    void init_all_rank();
    void init_master_rank();
    int get_cli(int argc, char **argv); // Returns int for status
    void init();

    // Setters
    void set_matrix(); // Reads from cli.f_matrix
    void set_matrix(c_matrix<INT, FLOAT> const &matrix);
    // The following was commented out in main.cpp, but let's declare it
    // and if its implementation is missing or problematic, we can address it.
    // It seems like it would call mat.read_matrix and then set_matrix(mat),
    // but mat.read_matrix modifies mat in place.
    // A better approach might be:
    // void set_matrix_from_file(std::string const &filename);
    // For now, let's keep the structure from main.cpp as close as possible
    // The original had: auto set_matrix(std::string const &filename)
    // which is problematic with template in .cpp. Let's make it void for now
    // and have it operate on the internal `mat` object.
    void set_matrix_from_file(std::string const &filename);

    void set_rhs();
    void set_icntl(int key, INT value);
    void set_cntl(int key, FLOAT value);
    void set_keep(int key, INT value); // Note: MUMPS keep array is INT based.
    void set_blr(const FLOAT &epsilon); // Changed from auto to const FLOAT&

    // Getters
    INT get_info(int key);     // Adjusted to match MUMPS array access (key-1) and type
    INT get_infog(int key);    // Adjusted to match MUMPS array access (key-1) and type
    FLOAT get_rinfo(int key);  // Adjusted to match MUMPS array access (key-1) and type
    FLOAT get_rinfog(int key); // Adjusted to match MUMPS array access (key-1) and type
    
    c_mumps_information get_all(); // Returns a copy of the maps object
    void dump();
};

#endif // C_MUMPS_SOLVER_HPP
