#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <functional>

#include "smumps_c.h"
#include "dmumps_c.h"
// add complex matrix ? 
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "../external/CLI11/include/CLI/CLI.hpp"

using namespace std ; 

#define USE_COMM_WORLD -987654; // MUMPS default comm
#define JOB_INIT -1 
#define JOB_END -2


enum MPI_execution_rank {MASTER, ALL, EXCLUDE_MASTER} ; 
// enable specific execution space thanks to template (todo?)
//
template <MPI_execution_rank RANK, typename FUNCTION>///, typename... ARGS>
void function_execution_space(FUNCTION func){//, ARGS... args){
	int rank ; 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if constexpr(RANK == MASTER){
		// execution on rank 0 only
		if (rank == 0 ){
			func();//args...);
			cout << "execution in rank " << rank << endl ; 
		}
	}
	else if constexpr(RANK == ALL){
                        func();//args...);
                        cout << "execution in rank " << rank << endl ;
	}
	else if constexpr(RANK == EXCLUDE_MASTER){
		if (rank != 0){
                        func();//args...);
                        cout << "execution in rank  " << rank << endl ;
		}
	}
}

int hey(){
	cout << "hey " << endl ;
       return 0 ; 	
}


// mumps structure
class c_mumps_information{
public:
	// 
	unordered_map<int, long long int> info ; 
	unordered_map<int, long long int> infog ; 
	unordered_map<int, double> rinfo; 
	unordered_map<int, double> rinfog;
};

template <typename INT, typename FLOAT>
class c_matrix {
public:
	INT nrows = 0 ; 
	INT ncols = 0 ; 
	vector<INT> rows, cols;
	vector<FLOAT> vals;

	void read_matrix(string const& filename){
		ifstream mfile ; 
		mfile.open(filename);
		fast_matrix_market::read_options options; 
		options.num_threads=1;
		fast_matrix_market::read_matrix_market_triplet(
			mfile, 
			nrows, 
			ncols,
			rows,
			cols,
			vals,
			options);
		return this;
	}
};

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT>
class c_mumps{
public:
	XMUMPS_STRUC_C mumps; 
	MPI_Comm comm = USE_COMM_WORLD ; 
	c_matrix<INT, FLOAT> mat ; 
	vector<FLOAT> rhs ; 

	//----------------- RUN 
	auto launch(){
		if constexpr(is_same_v<XMUMPS_STRUC_C, DMUMPS_STRUC_C>){
			dmumps_c(&mumps) ; 
		}
		else if constexpr(is_same_v<XMUMPS_STRUC_C, SMUMPS_STRUC_C>){
			smumps_c(&mumps);
		}
	}
	auto launch(int& job){
		mumps.job = job ; 
		launch() ; 
	}
	auto init_all_rank(){
		mumps.par = 1 ; 
		mumps.sym = 0 ; 
		mumps.comm_fortran = comm ; 
		launch(JOB_INIT) ; 
	}
	auto init_master_rank(){
                int rank ;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
                if (rank == 0){
                        for (int i = 0 ; i < mat.cols.size(); i++){
                                mat.cols[i] += 1;
                                mat.rows[i] += 1;
                        }
                        mumps.n = mat.rows ;
                        mumps.nnz = mat.vals.size();
                        mumps.irn = mat.rows.data();
                        mumps.jcn = mat.cols.data();
                        mumps.a = mat.vals.data();
                        mumps.rhs = rhs.data();
                }
	}

	auto init(){
		init_all_rank() ; 
		init_master_rank() ; 
	}
	///-------------- SETTERS
	auto set_matrix(c_matrix<INT, FLOAT> const& matrix){
		this->mat = matrix ; 
	}
	auto set_matrix(string const& filename){
//		this->matrix = c_matrix<INT, FLOAT>::read_matrix(filename);
		using cm = c_matrix<INT, FLOAT>;
		set_matrix(cm::read_matrix(filename));
	}
	auto set_rhs(){
		rhs (mat.nrows, 1.0) ; 
	}

	auto set_icntl(auto& key, auto& value){
		mumps.icntl[key-1] = static_cast<INT>(value) ; 
	}
	auto set_cntl(auto& key, auto& value){
		mumps.cntl[key-1] = static_cast<FLOAT>(value);
	}
	auto set_keep(auto& key, auto& value){
		mumps.keep[key-1] = static_cast<INT>(value);
	}
	auto set_blr(auto& epsilon){
		set_icntl(35,2);
		set_cntl(epsilon);
	}
	//---------------- GETTERS
	auto get_info(auto& key){
		return mumps.info[key-1] ; 
	}
	auto get_infog(auto& key){
		return mumps.infog[key-1];
	}
	auto get_rinfo(auto& key){
		return mumps.rinfo[key-1] ; 
	}
	auto get_rinfog(auto& key){
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



int main(int argc, char ** argv){
//	c_mumps<DMUMPS_STRUC_C, int, double> mumps {};
//
//	template <MPI_execution_rank RANK, typename FUNCTION>///, typename... ARGS>
//auto& function_execution_space(FUNCTION func){//, ARGS... args){
//
	MPI_Init(&argc, &argv) ; 
	function_execution_space<ALL>(hey) ; 	
	MPI_Finalize();

	return 0;
}





















