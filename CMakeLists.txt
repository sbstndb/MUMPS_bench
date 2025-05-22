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

find_package(nlohmann_json)

# Path to yout MUMPS installation. 
# Example for a Spack installation. Please edit this folowing line according to yout needs
set(MUMPS_DIR /home/sbstndbs/spack_for_samurai/opt/spack/linux-ubuntu24.10-skylake/gcc-14.2.0/mumps-5.7.3-2zyk6mffyj3vgipoznb336mixxx5rt3u)

include_directories(${MUMPS_DIR}/include)
link_directories(${MUMPS_DIR}/lib) 

set(SOURCES
	src/main.cpp	
)

set(CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=-qopenmp -O3")
add_executable(MUMPS_bench ${SOURCES})
target_link_libraries(MUMPS_bench OpenMP::OpenMP_CXX MPI::MPI_Fortran MPI::MPI_CXX mumps_common pord smumps dmumps fast_matrix_market)


