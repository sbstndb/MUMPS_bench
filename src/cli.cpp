#include "cli.hpp"
#include <CLI/CLI.hpp> // Should be included by cli.hpp, but good for clarity
#include <iostream>    // For std::cerr, std::endl
#include <vector>      // For std::vector
#include <string>      // For std::string

c_cli::c_cli() : app("MUMPS Benchmark"), f_matrix("../matrix/garon1/garon1.mtx"), f_logs("info.log"), epsilon(0.0) {
    // Constructor body
}

// The parse_key_value_option is now part of the header file as it's a template.
// No implementation needed in cli.cpp for parse_key_value_option.

int c_cli::get_cli(int argc, char **argv) {
    app.add_option("-m,--matrix", f_matrix, "Matrix path");
    app.add_option("-l,--log", f_logs, "Logfile name");

    app.add_option(
       "-i,--icntl",
       [this](CLI::results_t res) { // res is std::vector<std::string>
           // CLI::results_t is typedef for std::vector<std::string>
           return parse_key_value_option(res, this->icntl_params, "--icntl");
       },
       "ICNTL parameters key value (e.g., -i 7 2 -i 10 1)")
       ->type_name("INT INT")->expected(2)->take_all(); // CLI11 expects pairs, parse_key_value_option validates this.

    app.add_option(
       "-c,--cntl",
       [this](CLI::results_t res) { // res is std::vector<std::string>
            // CLI::results_t is typedef for std::vector<std::string>
           return parse_key_value_option(res, this->cntl_params, "--cntl");
       },
       "CNTL parameters key value (e.g., -c 1 0.01)")
       ->type_name("INT DOUBLE")->expected(2)->take_all(); // CLI11 expects pairs.

    app.add_option("-b,--blr", epsilon, "add BLR epsilon value (double)");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        // app.exit(e) prints the error message and exits.
        // It returns the exit code.
        return app.exit(e);
    }

    return 0; // Indicate success
}
