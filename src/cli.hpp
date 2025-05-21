#ifndef C_CLI_HPP
#define C_CLI_HPP

#include <CLI/CLI.hpp>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <type_traits> // Required for std::is_same_v

class c_cli {
public:
    CLI::App app;
    std::string f_matrix;
    std::string f_logs;
    std::map<int, int> icntl_params;
    std::map<int, double> cntl_params;
    double epsilon;

    c_cli();

    template <typename TValue>
    bool parse_key_value_option(
        const std::vector<std::string>& res_vector_of_strings, // Changed from CLI::results_t for clarity, it's std::vector<std::string>
        std::map<int, TValue>& target_map,
        const std::string& option_name) { // Added option_name for error reporting consistency
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
                    // Should not happen with current usage, but good for future-proofing
                    // static_assert dependent on TValue would require C++20 if false is always hit.
                    // For C++17, a more complex setup or simply runtime error is common.
                    std::cerr << "Error: Unsupported value type for parameter." << std::endl;
                    return false; // Or handle as appropriate
                }
                target_map[key] = value;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error parsing " << option_name << " parameter pair '"
                          << res_vector_of_strings[i] << "' '" << res_vector_of_strings[i+1]
                          << "': invalid number format. " << e.what() << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error parsing " << option_name << " parameter pair '"
                          << res_vector_of_strings[i] << "' '" << res_vector_of_strings[i+1]
                          << "': number out of range. " << e.what() << std::endl;
                return false;
            }
        }
        return true;
    }

    int get_cli(int argc, char **argv);
};

#endif // C_CLI_HPP
