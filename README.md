# MUMPS benchmark

This project provides a C++ wrapper for the MUMPS library desinged for benchmarking its performance with various inputs and configuration parameters. It leverages MPI for parallel execution and OpenMP for multithreading. 

## Features
- Flexible Matrix Input: Reads sparse matrices in Matrix Market format (.mtx)
- Command-Line intergace (CLI): Configurable via CLI11 for matrix path, log file, and various ICNTL or CNTL parameters
- Information logging: Dumps MUMPS INFO, INFOG, RINFO and RINFOG parameters to a specified log file

## Prerequisities
Before building and running the project, ensure you have the following installed:
- MPI Implementation
- MUMPS Library
- OpenMP
- CMake: Version 3.0 required
- Fortran Compiler required by MUMPS

## Building the project 
1. Clone the Repository:
```
git clone https://github.com/sbstndb/MUMPS_bench
cd MUMPS_bench
```
2. Ensure MUMPS is accessible:
3. The CMakeLists.txt file hardcoded the MUMPS installation directory. **YOU MUST** update the `MUMPS_DIR` in the `CMakeLists.txt` to point to your MUMPS installation path.
```
set(MUMPS_DIR /path/to/your/mumps/installation)
```
4. Build using CMake:
```
mkdir build
cd build
cmake ..
make -j 
```
This will create an executable named `MUMPS_bench` in the build directory. 

## Usage
The `MUMPS_bench` executabke can be run with MPI. Here is how to use its command-line options: 

```
mpirun -np <num_process> ./MUMPS_bench [options]
```
- -m, --marix <path_to_matrix.mtx>:
- -l, --log <logfile.log>
- -i, --icntl <KEY> <VALUE>
- -c, --cntl <KEY> <VALUE>
- -b, --blr <EPSILON>


## Examples
```
mpirun -np 4 ./MUMPS_bench
```

```
mpirun -np 4 ./MUMPS_bench -m /path/to/matrix.mtx -l mumps.log --icntl 35 2 --cntl 7 0.01
```

## TODO 
- [ ] Support for additional matrix formats (e.g. Harwell-Boeing, uncompressed CSR/CSC/COO)
- [ ] Error estimation: compute the error using the user specified RHS
- [x] Custom timers: add timers for each MUMPS phase
- [x] JSON Log output: Implement an option to export all collected MUMPS logs into a structured json format for easier parsing
- [ ] HPC metrics Collection: Integrate mechanisms to collect low-level hardware performance counters (e.g. CPU cycles, cache-misses, floating-point operations)
- [ ] `KEEP` parameters support: Add command-line options for `--keep` to allow users to set MUMPS experimental `KEEP` parameters, which control various internal options. 
