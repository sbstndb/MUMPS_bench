#include <iostream>
#include <mpi.h>
#include <vector> // Still needed for std::vector if used by main, or by included headers that don't bring it themselves.
                  // c_mumps_solver.hpp uses it for rhs.

// Custom class headers
#include "cli.hpp" 
// #include "mumps_log.hpp" // Not directly used in main, included by mumps_solver.hpp
// #include "matrix_utils.hpp" // Not directly used in main, included by mumps_solver.hpp
#include "mumps_solver.hpp" // This includes mumps_log.hpp, matrix_utils.hpp, cli.hpp, dmumps_c.h, smumps_c.h etc.

// Note: <functional>, <type_traits>, <dmumps_c.h>, <smumps_c.h>,
// <fast_matrix_market/fast_matrix_market.hpp>, and CLI11.hpp
// are no longer directly needed here as their functionalities are encapsulated
// in the new classes and included through mumps_solver.hpp or cli.hpp.

// Constants USE_COMM_WORLD, JOB_INIT, JOB_END are in mumps_solver.hpp

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  // MPI_COMM_WORLD is a macro typically defined in mpi.h
  // USE_COMM_WORLD was specific to the old c_mumps class, 
  // but c_mumps constructor now defaults to MPI_COMM_WORLD if not specified otherwise.
  // The mumps_solver.comm is initialized with USE_COMM_WORLD which is -987654.
  // This is fine as MUMPS interprets -987654 as MPI_COMM_WORLD.
  // For clarity, one might pass MPI_COMM_WORLD to c_mumps constructor if it was designed to take it.
  // For now, the existing logic in c_mumps handles this.
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Using DMUMPS_STRUC_C, int, double as the specific types for this benchmark
  c_mumps<DMUMPS_STRUC_C, int, double> mumps_solver{};

  // Parse Command Line Arguments
  if (mumps_solver.get_cli(argc, argv) != 0) {
    MPI_Finalize();
    return 1; // CLI parsing failed or --help was invoked.
  }

  // Set up and load the matrix based on CLI arguments
  // This is called by mumps_solver.set_matrix() which uses mumps_solver.cli.f_matrix
  mumps_solver.set_matrix(); // Loads matrix specified in CLI (or default)

  // Initialize MUMPS solver
  // This involves setting up MUMPS internal structures,
  // distributing matrix data if applicable (master rank handles matrix data),
  // and preparing for analysis/factorization.
  mumps_solver.init();

  // Apply ICNTL and CNTL parameters from CLI
  mumps_solver.apply_cli_params();

  if (rank == 0) {
    std::cout << "Starting MUMPS computation..." << std::endl;
  }

  // Perform MUMPS analysis, factorization, and solve (JOB = 6)
  mumps_solver.compute_all();

  if (rank == 0) {
    std::cout << "MUMPS Computation finished..." << std::endl;
  }

  // Dump MUMPS information and timing statistics to log file
  mumps_solver.dump();

  MPI_Finalize();
  return 0;
}
