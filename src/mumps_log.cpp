#include "mumps_log.hpp"
#include <mpi.h>      // For MPI_Comm_rank, MPI_COMM_WORLD
#include <fstream>    // For std::ofstream
#include <iostream>   // For std::cerr, std::endl

// Constructor
c_mumps_information::c_mumps_information() {
    // Default constructor, maps are initialized by their default constructors.
}

// Template method implementation must be in the header if it's to be used by
// different .cpp files without explicit instantiations.
// However, if write_map_to_file is ONLY used by write_maps_to_file within this SAME .cpp file,
// it can stay here. Given the problem description, it's a member of c_mumps_information
// and might be intended for wider use, suggesting it should be in the header.
// For now, placing it here as per typical .cpp structure for non-template methods.
// If linker errors occur later, this is a candidate to move to .hpp.
// **Correction**: The prompt asks for write_map_to_file to be in .cpp.
// This will work as long as it's only called by methods within c_mumps_information.cpp
// OR if specific template instantiations are provided.
// The typical way it was in main.cpp (as a template method within a class, called by another method of the same class)
// is fine. Let's stick to the structure from main.cpp.

template <typename K, typename V>
void c_mumps_information::write_map_to_file(std::ostream &file,
                                             const std::unordered_map<K, V> &map,
                                             const std::string &type) {
    for (const auto &pair : map) {
        file << type << pair.first << ": " << pair.second << std::endl;
    }
}

// Explicit instantiations for the template method might be needed if called from outside this file,
// or move the template definition to the header.
// Since it's called by write_maps_to_file in this same file, it should be fine.
template void c_mumps_information::write_map_to_file<int, long long int>(std::ostream &file, const std::unordered_map<int, long long int> &map, const std::string &type);
template void c_mumps_information::write_map_to_file<int, double>(std::ostream &file, const std::unordered_map<int, double> &map, const std::string &type);


void c_mumps_information::write_maps_to_file(std::string const &filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        // only on master rank
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: could not open log file " << filename << std::endl; // Corrected "Errror" to "Error"
        } else {
            // As per original main.cpp, only infog and rinfog are written.
            // The commented out lines for info and rinfo are preserved.
            write_map_to_file(file, infog, "infog");
            write_map_to_file(file, rinfog, "rinfog");
            // write_map_to_file(file, info, "info");
            // write_map_to_file(file, rinfo, "rinfo");
        }
    }
}
