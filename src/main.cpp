#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "smumps_c.h"
#include "dmumps_c.h"
// add complex matrix ? 
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "../external/CLI11/include/CLI/CLI.hpp"

using namespace std ; 

#define USE_COMM_WORLD -987654; // MUMPS default comm
#define JOB_INIT -1; 
#define JOB_END -2;

// mumps structure

class c_mumps_information{
public:
	// 
	unordered_map<int, long long int> info ; 
	unordered_map<int, long long int> infog ; 
	unordered_map<int, double> rinfo; 
	unordered_map<int, double> rinfog;
};

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
class c_mumps{
public:
	XMUMPS_STRUC_C mumps; 
	MPI_Comm comm = USE_COMM_WORLD ; 

	//-----------------------
	auto launch(){
		if constexpr(is_same_v<XMUMPS_STRUC_C, DMUMPS_STRUC_C>){
			dmumps_c(&mumps) ; 
		}
		else if constexpr(is_same_v<XMUMPS_STRUC_C, SMUMPS_STRUC_C>){
			smumps_c(&mumps);
		}
	}
	///-------------- SETTERS
	auto set_icntl(auto key, auto value){
		mumps.icntl[key-1] = static_cast<INT>(value) ; 
	}
	auto set_cntl(auto key, auto value){
		mumps.cntl[key-1] = static_cast<FLOAT>(value);
	}
	auto set_keep(auto key, auto value){
		mumps.keep[key-1] = static_cast<INT>(value);
	}
	auto set_blr(auto epsilon){
		set_icntl(35,2);
		set_cntl(epsilon);
	}
	//---------------- GETTERS
	auto get_info(auto key){
		return mumps.info[key-1] ; 
	}
	auto get_infog(auto key){
		return mumps.infog[key-1];
	}
	auto get_rinfo(auto key){
		return mumps.rinfo[key-1] ; 
	}
	auto get_rinfog(auto key){
		return mumps.rinfog[key-1];
	}	
	auto get_all(){
		// fortran's mumps define arrays of these sizes, some are useless but lets get them for futureproofness
		static int max_info=80 ; 
		static int max_infog=80;
		static int max_rinfo = 40;
		static int max_rinfog=40;
		c_mumps_information maps ;
		// here we use static_cast<long long int> but thast BAD !!
		// we should use template for true bitwise getter
		for (unsigned int i = 0 ; i < max_info ; i++){
			maps.info[i] = static_cast<long long int>(mumps.info[i]);
                        maps.infog[i] = static_cast<long long int>(mumps.infog[i]);			
		}
                for (unsigned int i = 0 ; i < max_info ; i++){
                        maps.rinfo[i] = static_cast<double>(mumps.rinfo[i]);
                        maps.rinfog[i] = static_cast<double>(mumps.rinfog[i]);
                }
		return maps ; 

	}


};



int main(){
	c_mumps<DMUMPS_STRUC_C, int, double> mumps {};
	return 0;
}





















