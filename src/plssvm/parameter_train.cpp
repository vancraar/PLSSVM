/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter_train.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>     // stderr
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <string>     // std::string
#include <utility>    // std::move

namespace plssvm {

template <typename T>
parameter_train<T>::parameter_train(std::string p_input_filename) {
    base_type::input_filename = std::move(p_input_filename);
    base_type::model_filename = base_type::model_name_from_input();

    base_type::parse_train_file(input_filename);
}

template <typename T>
parameter_train<T>::parameter_train(int argc, char **argv) {
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("training_set_file [model_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear: u'*v\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<decltype(kernel)>()->default_value(fmt::format("{}", detail::to_underlying(kernel))))
            ("d,degree", "set degree in kernel function", cxxopts::value<decltype(degree)>()->default_value(fmt::format("{}", degree)))
            ("g,gamma", "set gamma in kernel function (default: 1 / num_features)", cxxopts::value<decltype(gamma)>())
            ("r,coef0", "set coef0 in kernel function", cxxopts::value<decltype(coef0)>()->default_value(fmt::format("{}", coef0)))
            ("c,cost", "set the parameter C", cxxopts::value<decltype(cost)>()->default_value(fmt::format("{}", cost)))
            ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<decltype(epsilon)>()->default_value(fmt::format("{}", epsilon)))
            ("b,backend", "choose the backend: openmp|cuda|opencl|sycl", cxxopts::value<decltype(backend)>()->default_value(detail::as_lower_case(fmt::format("{}", backend))))
            ("p,target_platform", "choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel", cxxopts::value<decltype(target)>()->default_value(detail::as_lower_case(fmt::format("{}", target))))
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info)->default_value(fmt::format("{}", !print_info)))
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("input", "", cxxopts::value<decltype(input_filename)>(), "training_set_file")
            ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "model" });
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        fmt::print("{}\n{}\n", e.what(), options.help());
        std::exit(EXIT_FAILURE);
    }

    // print help message and exit
    if (result.count("help")) {
        fmt::print("{}", options.help());
        std::exit(EXIT_SUCCESS);
    }

    // parse kernel_type and cast the value to the respective enum
    kernel = result["kernel_type"].as<decltype(kernel)>();

    // parse degree
    degree = result["degree"].as<decltype(degree)>();

    // parse gamma
    if (result.count("gamma")) {
        gamma = result["gamma"].as<decltype(gamma)>();
        if (gamma == decltype(gamma){ 0.0 }) {
            fmt::print(stderr, "gamma = 0.0 is not allowed, it doesnt make any sense!\n");
            fmt::print("{}", options.help());
            std::exit(EXIT_FAILURE);
        }
    } else {
        gamma = decltype(gamma){ 0.0 };
    }

    // parse coef0
    coef0 = result["coef0"].as<decltype(coef0)>();

    // parse cost
    cost = result["cost"].as<decltype(cost)>();

    // parse epsilon
    epsilon = result["epsilon"].as<decltype(epsilon)>();

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

    // parse print info
    print_info = !print_info;

    // parse input data filename
    if (!result.count("input")) {
        fmt::print(stderr, "Error missing input file!");
        fmt::print("{}", options.help());
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.count("model")) {
        model_filename = result["model"].as<decltype(model_filename)>();
    } else {
        model_filename = base_type::model_name_from_input();
    }

    base_type::parse_train_file(input_filename);
}

// explicitly instantiate template class
template class parameter_train<float>;
template class parameter_train<double>;

}  // namespace plssvm