cmake_minimum_required(VERSION 3.0)
project(MUMPS_bench)

enable_language(Fortran)
enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP COMPONENTS Fortran CXX REQUIRED)
find_package(MPI COMPONENTS Fortran CXX REQUIRED)

option(BUILD_SHARED_LIBS ON)

add_subdirectory(external/mumps)
add_subdirectory(external/CLI11)
add_subdirectory(external/fast_matrix_market)


set(SOURCES
	src/main.cpp

	
)

set(CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=-qopenmp")
add_executable(MUMPS_bench ${SOURCES})
target_link_libraries(MUMPS_bench MPI::MPI_Fortran MPI::MPI_CXX mumps_common pord smumps dmumps fast_matrix_market)


