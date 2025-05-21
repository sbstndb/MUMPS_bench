cmake_minimum_required(VERSION 3.0)
project(MUMPS_bench)

enable_language(Fortran)
enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP COMPONENTS Fortran CXX REQUIRED)
find_package(MPI COMPONENTS Fortran CXX REQUIRED)
option(BUILD_SHARED_LIBS ON)

add_subdirectory(external/CLI11)
add_subdirectory(external/fast_matrix_market)


# Chemins vers ton installation MUMPS
set(MUMPS_DIR /home/sbstndbs/spack_for_samurai/opt/spack/linux-ubuntu24.10-skylake/gcc-14.2.0/mumps-5.7.3-2zyk6mffyj3vgipoznb336mixxx5rt3u)

include_directories(${MUMPS_DIR}/include)
link_directories(${MUMPS_DIR}/lib)  # ou lib64 si c’est là



set(SOURCES
	src/main.cpp
	src/cli.cpp
	src/mumps_log.cpp
	src/matrix_utils.cpp
	src/mumps_solver.cpp
)

set(CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=-qopenmp")
add_executable(MUMPS_bench ${SOURCES})
target_link_libraries(MUMPS_bench OpenMP::OpenMP_CXX MPI::MPI_Fortran MPI::MPI_CXX mumps_common pord smumps dmumps fast_matrix_market CLI11::CLI11)


