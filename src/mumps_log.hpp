#ifndef C_MUMPS_INFORMATION_HPP
#define C_MUMPS_INFORMATION_HPP

#include <string>
#include <vector> // Though not directly used by c_mumps_information, it's in the list.
#include <map>    // Though not directly used by c_mumps_information, it's in the list. (unordered_map is used)
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <mpi.h>

class c_mumps_information {
public:
    std::unordered_map<int, long long int> info;
    std::unordered_map<int, long long int> infog;
    std::unordered_map<int, double> rinfo;
    std::unordered_map<int, double> rinfog;

    c_mumps_information();

    template <typename K, typename V>
    void write_map_to_file(std::ostream &file,
                           const std::unordered_map<K, V> &map,
                           const std::string &type);

    void write_maps_to_file(std::string const &filename);
};

#endif // C_MUMPS_INFORMATION_HPP
