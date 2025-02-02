/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/exceptions/exceptions.hpp"

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include "fmt/core.h"  // fmt::format

#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm {

exception::exception(const std::string &msg, const std::string_view class_name, source_location loc) :
    std::runtime_error{ msg }, class_name_{ class_name }, loc_{ loc } {}

const source_location &exception::loc() const noexcept { return loc_; }

std::string exception::what_with_loc() const {
    return fmt::format(
        "{}\n"
        "{} thrown:\n"
        "  in file      {}\n"
        "  in function  {}\n"
        "  @ line       {}",
        this->what(),
        class_name_,
        loc_.file_name(),
        loc_.function_name(),
        loc_.line());
}

file_not_found_exception::file_not_found_exception(const std::string &msg, source_location loc) :
    exception{ msg, "file_not_found_exception", loc } {}

invalid_file_format_exception::invalid_file_format_exception(const std::string &msg, source_location loc) :
    exception{ msg, "invalid_file_format_exception", loc } {}

unsupported_backend_exception::unsupported_backend_exception(const std::string &msg, source_location loc) :
    exception{ msg, "unsupported_backend_exception", loc } {}

unsupported_kernel_type_exception::unsupported_kernel_type_exception(const std::string &msg, source_location loc) :
    exception{ msg, "unsupported_kernel_type_exception", loc } {}

}  // namespace plssvm