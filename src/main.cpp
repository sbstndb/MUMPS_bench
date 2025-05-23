#include <functional>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include <type_traits>

#include "../external/CLI11/include/CLI/CLI.hpp"
#include <dmumps_c.h>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <smumps_c.h>


#include "nlohmann/json.hpp"

using json = nlohmann::json;

constexpr int USE_COMM_WORLD = -987654;
constexpr int JOB_INIT = -1;
constexpr int JOB_END = -2;

constexpr int MUMPS_INFO_SIZE = 80;
constexpr int MUMPS_RINFO_SIZE = 40; 


class c_cli {
public:
  CLI::App app{"MUMPS Benchmark"};
  std::string f_matrix = "../matrix/garon1/garon1.mtx";
  //	std::string f_matrix = "../matrix/bcsstm12/bcsstm12.mtx";
  std::string f_logs = "info.json";

  std::map<int, int> icntl_params;
  std::map<int, double> cntl_params;

  double epsilon;

  c_cli() = default;


    template <typename TValue>
    bool parse_key_value_option(
        CLI::results_t res_vector_of_strings,
        std::map<int, TValue>& target_map,
        const std::string& option_name)
    {
        if (res_vector_of_strings.size() % 2 != 0) {
            std::cerr << "Error: " << option_name << " expects arguments in pairs (key-value), "
                      << "but the total number of arguments received ("
                      << res_vector_of_strings.size() << ") is not even." << std::endl;
            return false;
        }
        for (size_t i = 0; i < res_vector_of_strings.size(); i += 2) {
            try {
                int key = std::stoi(res_vector_of_strings[i]);
                TValue value;
                if constexpr (std::is_same_v<TValue, int>) {
                    value = std::stoi(res_vector_of_strings[i + 1]);
                } else if constexpr (std::is_same_v<TValue, double>) {
                    value = std::stod(res_vector_of_strings[i + 1]);
                } else {
                    // Should not happen with current usage
                    static_assert(std::is_same_v<TValue, int> || std::is_same_v<TValue, double>, "Unsupported value type");
                }
                target_map[key] = value;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error parsing " << option_name << " parameter pair "
                          << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                          << ": invalid number format. " << e.what() << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error parsing " << option_name << " parameter pair "
                          << res_vector_of_strings[i] << " " << res_vector_of_strings[i+1]
                          << ": number out of range. " << e.what() << std::endl;
                return false;
            }
        }
        return true;
    }

	
    int get_cli(int argc, char **argv) {
    app.add_option("-m,--matrix", f_matrix, "Matrix path");
    app.add_option("-l,--log", f_logs, "Logfile name");

        app.add_option(
           "-i,--icntl",
           [this](CLI::results_t res) { return parse_key_value_option(res, this->icntl_params, "--icntl"); },
           "ICNTL parameters...")
           ->type_name("INT INT")->expected(2)->take_all();

        app.add_option(
           "-c,--cntl",
           [this](CLI::results_t res) { return parse_key_value_option(res, this->cntl_params, "--cntl"); },
           "CNTL parameters...")
           ->type_name("INT DOUBLE")->expected(2)->take_all();
         

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

class c_mumps_information {
public:
  //
  std::unordered_map<int, long long int> info;
  std::unordered_map<int, long long int> infog;
  std::unordered_map<int, double> rinfo;
  std::unordered_map<int, double> rinfog;

  double time_analysis_s = 0.0;
  double time_factorize_s = 0.0;
  double time_solve_s = 0.0;
  double time_total_s = 0.0; // Temps total des 3 phases


  c_mumps_information() = default;

  template <typename K, typename V>
  void write_map_to_file(std::ostream &file,
                         const std::unordered_map<K, V> &map,
                         const std::string &type) {
    for (const auto &pair : map) {
      file << type << pair.first << ": " << pair.second << std::endl;
    }
  }
  // ISSUE : MUMPS can be launched in a MPI way hence r____ types are per rank
  // values then we need to save these datas per rank I suggest to make a
  // specific write_map for master rank and another for full rank with keynames
  // like rinfog23r<rank>: ... at the moment
  void write_global_maps_to_file(std::string const &filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      // only on master rank
      std::ofstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Errror : cound not open log file" << filename
                  << std::endl;
      } else {
        write_map_to_file(file, infog, "infog");
        write_map_to_file(file, rinfog, "rinfog");
      }
    }
  }

  void write_local_maps_to_file(std::string const &filename, int rank_id) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      // only on master rank
      std::ofstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Errror : cound not open log file" << filename
                  << std::endl;
      } else {
	std::string info_prefix = "info_r" + std::to_string(rank_id) + "_" ; 
        std::string rinfo_prefix = "rinfo_r" + std::to_string(rank_id) + "_" ;
	
        write_map_to_file(file, info, info_prefix);
        write_map_to_file(file, rinfo, rinfo_prefix);
      }
    }
  }

  template <typename K, typename V>
  json map_to_json(const std::unordered_map<K, V>& map_data) const {
	json j_map = json::object();
	for (const auto& pair: map_data){
		j_map[std::to_string(pair.first)] = pair.second ; 
	}
	return j_map ; 
  }

  json to_json_global_only() const {
	json j; 
	j["infog"] = map_to_json(infog) ; 
        j["rinfog"] = map_to_json(rinfog) ;

	j["elapsed_analysis"] = time_analysis_s; 
        j["elapsed_factorize"] = time_factorize_s;
        j["elapsed_solve"] = time_solve_s;
        j["elapsed_total"] = time_total_s;
	return j ; 
  }


};



template <typename INT, typename FLOAT> class c_matrix {
public:
  INT nrows = 0;
  INT ncols = 0;
  std::vector<INT> rows, cols;
  std::vector<FLOAT> vals;

  c_matrix() = default;

  auto read_matrix(std::string const &filename) {
    std::ifstream mfile;
    mfile.open(filename);
    fast_matrix_market::read_options options;
    options.num_threads = 1;
    fast_matrix_market::read_matrix_market_triplet(mfile, nrows, ncols, rows,
                                                   cols, vals, options);
    return this;
  }
};

template <typename XMUMPS_STRUC_C, typename INT, typename FLOAT> class c_mumps {
public:
  XMUMPS_STRUC_C mumps;
  MPI_Comm comm = USE_COMM_WORLD;
  c_matrix<INT, FLOAT> mat;
  std::vector<FLOAT> rhs;
  c_mumps_information maps;
  std::vector<c_mumps_information> all_ranks_local_maps;
  c_cli cli;

  std::chrono::duration<double> duration_analysis;
  std::chrono::duration<double> duration_factorize;
  std::chrono::duration<double> duration_solve;
  std::chrono::duration<double> duration_all;

  auto apply_cli_params() {
    for (const auto &param : cli.icntl_params) {
      set_icntl(param.first, param.second);
    }
    for (const auto &param : cli.cntl_params) {
      set_cntl(param.first, param.second);
    }
    if (cli.epsilon != 0.0) {
      set_cntl(7, cli.epsilon);
      set_icntl(35, 2);
    }
  }

  auto launch() {
    if constexpr (std::is_same_v<XMUMPS_STRUC_C, DMUMPS_STRUC_C>) {
      dmumps_c(&mumps);
    } else if constexpr (std::is_same_v<XMUMPS_STRUC_C, SMUMPS_STRUC_C>) {
      smumps_c(&mumps);
    }
  }

  auto launch(int job) {
    mumps.job = job;
    launch();
  }
  auto analysis() { 
	  auto start = std::chrono::high_resolution_clock::now(); 
	  launch(1); 
	  auto end = std::chrono::high_resolution_clock::now();
	duration_analysis = end - start;
  }
  auto factorize() { 
          auto start = std::chrono::high_resolution_clock::now();	  
	  launch(2);
          auto end = std::chrono::high_resolution_clock::now();
        duration_factorize = end - start	  ;
  }
  auto solve() { 
          auto start = std::chrono::high_resolution_clock::now(); 
	  launch(3);
          auto end = std::chrono::high_resolution_clock::now();
	  // icntl11 to 2 to compute the error etc
	  auto old_value = get_icntl(11); 
	  set_icntl(11, 2);
	  launch(3) ; 
	  set_icntl(11, old_value);
        duration_solve = end - start	  ;
  }
  auto compute_all() { 
          auto start = std::chrono::high_resolution_clock::now();
	  analysis();
	  factorize();
	  solve();
          auto end = std::chrono::high_resolution_clock::now();
        duration_all = end - start	  ;
  }
  auto init_all_rank() {
    mumps.par = 1;
    mumps.sym = 0;
    mumps.comm_fortran = comm;
    launch(JOB_INIT);
  }
  auto init_master_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      for (int i = 0; i < mat.cols.size(); i++) {
        mat.cols[i] += 1;
        mat.rows[i] += 1;
      }
      mumps.n = mat.nrows;
      mumps.nnz = mat.vals.size();
      mumps.irn = mat.rows.data();
      mumps.jcn = mat.cols.data();
      mumps.a = mat.vals.data();
      set_rhs();
      mumps.rhs = rhs.data();
    }
  }

  auto get_cli(int argc, char **argv) { return cli.get_cli(argc, argv); }

  auto init() {
    init_all_rank();
    init_master_rank();
  }
  ///-------------- SETTERS
  auto set_matrix() {
    mat.read_matrix(cli.f_matrix);
    set_matrix(mat);
  }
  auto set_matrix(c_matrix<INT, FLOAT> const &matrix) { this->mat = matrix; }
  auto set_matrix(std::string const &filename) {
    //		this->matrix = c_matrix<INT, FLOAT>::read_matrix(filename);
    using cm = c_matrix<INT, FLOAT>;
    set_matrix(cm::read_matrix(filename));
  }
  auto set_rhs() { rhs.assign(mat.nrows, 1.0); }

  auto set_icntl(int key, INT value) { mumps.icntl[key - 1] = value; }
  auto set_cntl(int key, FLOAT value) { mumps.cntl[key - 1] = value; }
  auto set_keep(int key, INT value) { mumps.keep[key - 1] = value; }
  auto set_blr(auto &epsilon) {
    set_icntl(35, 2);
    set_cntl(epsilon);
  }
  //---------------- GETTERS
  auto get_icntl(auto key) { return mumps.icntl[key - 1]; }
  auto get_info(auto &key) { return mumps.info[key - 1]; }
  auto get_infog(auto &key) { return mumps.infog[key - 1]; }
  auto get_rinfo(auto &key) { return mumps.rinfo[key - 1]; }
  auto get_rinfog(auto &key) { return mumps.rinfog[key - 1]; }
  auto get_all() {
    // fortran's mumps define arrays of these sizes, some are useless but lets
    // get them for futureproofness
    static int max_info = 80;
    static int max_infog = 80;
    static int max_rinfo = 40;
    static int max_rinfog = 40;
    c_mumps_information maps;
    // here we use static_cast<long long int> but thast BAD !!
    // we should use template for true bitwise getter
    for (unsigned int i = 0; i < MUMPS_INFO_SIZE; i++) {
      maps.info[i + 1] = static_cast<long long int>(mumps.info[i]);
      maps.infog[i + 1] = static_cast<long long int>(mumps.infog[i]);
    }
    for (unsigned int i = 0; i < MUMPS_INFO_SIZE; i++) {
      maps.rinfo[i + 1] = static_cast<double>(mumps.rinfo[i]);
      maps.rinfog[i + 1] = static_cast<double>(mumps.rinfog[i]);
    }
	maps.time_analysis_s = duration_analysis.count(); 
	
        maps.time_analysis_s = duration_analysis.count();
        maps.time_factorize_s = duration_factorize.count();
        maps.time_solve_s = duration_solve.count();
        maps.time_total_s = duration_analysis.count() + duration_factorize.count() + duration_solve.count();
    return maps;
  }



  auto dump() {
    maps = get_all();
    // add MPI transfer of info.rinfo data
    //
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    if (rank ==0){

	    json final_output = maps.to_json_global_only() ;  
	    std::ofstream file(cli.f_logs) ; 
	    file << final_output.dump(4) ; 
   }
    

//    maps.write_global_maps_to_file(cli.f_logs);
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank ; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  c_mumps<DMUMPS_STRUC_C, int, double> mumps_solver{};
  if (mumps_solver.get_cli(argc, argv) != 0){
	MPI_Finalize() ; 
	return 1; 
  }
  c_matrix<int, double> matrix;
  mumps_solver.set_matrix();
  mumps_solver.init();
  mumps_solver.apply_cli_params();


  if (rank == 0){
	std::cout << "Starting MUMPS computation..." << std::endl ; 
  }
  mumps_solver.compute_all();
  if (rank == 0){
	std::cout << "MUMPS Computation finished..." << std::endl ; 
  }
  mumps_solver.dump();

  MPI_Finalize();

  return 0;
}
