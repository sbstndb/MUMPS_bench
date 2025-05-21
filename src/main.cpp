#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <functional>

#include <type_traits>

#include <smumps_c.h>
#include <dmumps_c.h>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "../external/CLI11/include/CLI/CLI.hpp"


constexpr int USE_COMM_WORLD = -987654;
constexpr int JOB_INIT = -1 ; 
constexpr int JOB_END = -2 ; 

//#define USE_COMM_WORLD -987654; // MUMPS default comm
//#define JOB_INIT -1 
//#define JOB_END -2


class c_cli{
public:
	CLI::App app{"MUMPS Benchmark"};
	std::string f_matrix = "../matrix/garon1/garon1.mtx";	
//	std::string f_matrix = "../matrix/bcsstm12/bcsstm12.mtx";
	std::string f_logs = "info.log";

	std::map<int, int> icntl_params ; 
	std::map<int, double> cntl_params;

	double epsilon ; 

	c_cli() = default ; 


 int get_cli(int argc, char** argv) {
        app.add_option("-m,--matrix", f_matrix, "Matrix path");
        app.add_option("-l,--log", f_logs, "Logfile name");

app.add_option("-i,--icntl",
                       [this](CLI::results_t res_vector_of_strings){
                           // This check ensures the total number of arguments is a multiple of 2.
                           // It's useful even with expected(2) as a safety check.
                           if (res_vector_of_strings.size() % 2 != 0) {
                               std::cerr << "Error: --icntl expects arguments in pairs (key-value), but the total number of arguments received ("
                                         << res_vector_of_strings.size() << ") is not even." << std::endl;
                               return false; // Indicate validation failure
                           }

                           // Process arguments in pairs, as they are accumulated from all uses
                           for (size_t i = 0; i < res_vector_of_strings.size(); i += 2) {
                               try {
                                   int key = std::stoi(res_vector_of_strings[i]);
                                   int value = std::stoi(res_vector_of_strings[i+1]);

                                   // Store the parsed key-value pair
                                   this->icntl_params[key] = value;

                               } catch (const std::invalid_argument& e) {
                                   // Error parsing the pair starting at index i
                                   std::cerr << "Error parsing ICNTL parameter pair --icntl "
                                             << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                                             << " (from argument position " << i + 1 << "/" << i + 2 << "): invalid number format. " << e.what() << std::endl;
                                   return false; // Indicate validation failure
                               } catch (const std::out_of_range& e) {
                                    // Out of range could apply to key or value
                                   std::cerr << "Error parsing ICNTL parameter pair --icntl "
                                             << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                                              << " (from argument position " << i + 1 << "/" << i + 2 << "): number out of integer range. " << e.what() << std::endl;
                                    return false; // Indicate validation failure
                               }
                           }

                           // If the loop completes without returning false, all pairs were parsed successfully
                           return true;
                       },
                       "ICNTL parameters as key-value pairs (int key, int value). Repeat option for multiple pairs (e.g., -i 35 2 -i 36 1). Arguments from multiple uses are combined.")
           ->type_name("INT INT") // Still good for help text
           ->expected(2)        // <-- Keep this: Each time -i or --icntl appears, expect 2 arguments
//		->allow_multiple_occurrences();
->take_all();


app.add_option("-c,--cntl",
                       [this](CLI::results_t res_vector_of_strings){
                           // This check ensures the total number of arguments is a multiple of 2.
                           // It's useful even with expected(2) as a safety check.
                           if (res_vector_of_strings.size() % 2 != 0) {
                               std::cerr << "Error: --cntl expects arguments in pairs (key-value), but the total number of arguments received ("
                                         << res_vector_of_strings.size() << ") is not even." << std::endl;
                               return false; // Indicate validation failure
                           }

                           // Process arguments in pairs, as they are accumulated from all uses
                           for (size_t i = 0; i < res_vector_of_strings.size(); i += 2) {
                               try {
                                   int key = std::stoi(res_vector_of_strings[i]);
                                   double value = std::stod(res_vector_of_strings[i+1]);

                                   // Store the parsed key-value pair
                                   this->cntl_params[key] = value;

                               } catch (const std::invalid_argument& e) {
                                   // Error parsing the pair starting at index i
                                   std::cerr << "Error parsing CNTL parameter pair --cntl "
                                             << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                                             << " (from argument position " << i + 1 << "/" << i + 2 << "): invalid number format. " << e.what() << std::endl;
                                   return false; // Indicate validation failure
                               } catch (const std::out_of_range& e) {
                                    // Out of range could apply to key or value
                                   std::cerr << "Error parsing ICNTL parameter pair --cntl "
                                             << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                                              << " (from argument position " << i + 1 << "/" << i + 2 << "): number out of integer range. " << e.what() << std::endl;
                                    return false; // Indicate validation failure
                               }
                           }

                           // If the loop completes without returning false, all pairs were parsed successfully
                           return true;
                       },
                       "CNTL parameters as key-value pairs (int key, double value). Repeat option for multiple pairs (e.g., -i 7 0.001). Arguments from multiple uses are combined.")
           ->type_name("INT DOUBLE") // Still good for help text
           ->expected(2)        // <-- Keep this: Each time -i or --cntl appears, expect 2 arguments
//              ->allow_multiple_occurrences();
->take_all();




        app.add_option("-b,--blr", epsilon, "add BLR epsilon value");


        try {
            app.parse(argc, argv);

        } catch (const CLI::ParseError &e) {
            app.exit(e);
            return 1; // Indicate failure
        }

        return 0; // Indicate success
    }

};



class c_mumps_information{
public:
	// 
	std::unordered_map<int, long long int> info ; 
	std::unordered_map<int, long long int> infog ; 
	std::unordered_map<int, double> rinfo; 
	std::unordered_map<int, double> rinfog;

	c_mumps_information() = default; 

	template <typename K, typename V>
	void write_map_to_file(std::ostream& file, const std::unordered_map<K, V>& map, const std::string& type){
		for (const auto& pair : map){
			file << type << pair.first << ": " << pair.second << std::endl ; 
		}
	}
	// ISSUE : MUMPS can be launched in a MPI way hence r____ types are per rank values
	// then we need to save these datas per rank
	// I suggest to make a specific write_map for master rank and another for full rank with 
	// keynames like rinfog23r<rank>: ... at the moment
	void write_maps_to_file(std::string const& filename){
                int rank ;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
                if (rank == 0){
			// only on master rank 
			std::ofstream file(filename);
			if (!file.is_open()){
				std::cerr << "Errror : cound not open log file" << filename << std::endl ; 
			}
			else {
		                write_map_to_file(file, infog, "infog");
		                write_map_to_file(file, rinfog, "rinfog");
	                        //write_map_to_file(file, info, "info");
	                        //write_map_to_file(file, rinfo, "rinfo");
			}
		}
	}
};

template <typename INT, typename FLOAT>
class c_matrix {
public:
	INT nrows = 0 ; 
	INT ncols = 0 ; 
	std::vector<INT> rows, cols;
	std::vector<FLOAT> vals;

	c_matrix() = default ; 

	auto read_matrix(std::string const& filename){
		std::ifstream mfile ; 
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
	std::vector<FLOAT> rhs ; 
	c_mumps_information maps ; 
	c_cli cli ; 

	auto apply_cli_params(){
		for (const auto& param : cli.icntl_params){
			set_icntl(param.first, param.second) ; 
		
		}
                for (const auto& param : cli.cntl_params){
                        set_cntl(param.first, param.second) ;
                }
		if (cli.epsilon != 0.0){
			set_cntl(7, cli.epsilon); 
			set_icntl(35, 2); 
		}
	}

	auto launch(){
		if constexpr(std::is_same_v<XMUMPS_STRUC_C, DMUMPS_STRUC_C>){
			dmumps_c(&mumps) ; 
		}
		else if constexpr(std::is_same_v<XMUMPS_STRUC_C, SMUMPS_STRUC_C>){
			smumps_c(&mumps);
		}
	}

        auto launch(int job){
                mumps.job = job ;
                launch() ;
        }
	auto analysis(){
		launch(1);
	}
	auto factorize(){
		launch(2);
	}
	auto solve(){
		launch(3);
	}
	auto compute_all(){
		launch(6);
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
                        mumps.n = mat.nrows ;
                        mumps.nnz = mat.vals.size();
                        mumps.irn = mat.rows.data();
                        mumps.jcn = mat.cols.data();
                        mumps.a = mat.vals.data();
			set_rhs();
                        mumps.rhs = rhs.data();
                }
	}

	auto get_cli(int argc, char** argv){
		cli.get_cli(argc, argv);
	}

	auto init(){
		init_all_rank() ; 
		init_master_rank();

	}
	///-------------- SETTERS
	auto set_matrix(){
		mat.read_matrix(cli.f_matrix);
		set_matrix(mat);
	}
	auto set_matrix(c_matrix<INT, FLOAT> const& matrix){
		this->mat = matrix ; 
	}
	auto set_matrix(std::string const& filename){
//		this->matrix = c_matrix<INT, FLOAT>::read_matrix(filename);
		using cm = c_matrix<INT, FLOAT>;
		set_matrix(cm::read_matrix(filename));
	}
	auto set_rhs(){
		rhs.assign(mat.nrows, 1.0) ; 
	}

	auto set_icntl(int key, INT value){
		mumps.icntl[key-1] = value ; 
	}
	auto set_cntl(int key, FLOAT value){
		mumps.cntl[key-1] = value;
	}
	auto set_keep(int key, INT value){
		mumps.keep[key-1] = value;
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
			maps.info[i+1] = static_cast<long long int>(mumps.info[i]);
                        maps.infog[i+1] = static_cast<long long int>(mumps.infog[i]);			
		}
                for (unsigned int i = 0 ; i < max_info ; i++){
                        maps.rinfo[i+1] = static_cast<double>(mumps.rinfo[i]);
                        maps.rinfog[i+1] = static_cast<double>(mumps.rinfog[i]);
                }
		return maps ; 
	}

	auto dump(){
		maps = get_all() ; 
		maps.write_maps_to_file(cli.f_logs);
	}
};



int main(int argc, char ** argv){
	MPI_Init(&argc, &argv) ; 

        c_mumps<DMUMPS_STRUC_C, int, double> mumps {};
	mumps.get_cli(argc, argv);
	c_matrix<int, double> matrix ;
	mumps.set_matrix();
	mumps.init() ; 
        mumps.apply_cli_params();

	mumps.compute_all();
	mumps.dump();

	MPI_Finalize();

	return 0;
}





















