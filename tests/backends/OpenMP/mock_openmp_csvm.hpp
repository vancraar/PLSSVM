/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the OpenMP backend.
 */

#pragma once

#include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#include "plssvm/parameter.hpp"             // plssvm::parameter

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the OpenMP CSVM.
 * @tparam T the type of the data
 */
template <typename T>
class mock_openmp_csvm : public plssvm::openmp::csvm<T> {
    using base_type = plssvm::openmp::csvm<T>;

  public:
    using real_type = typename base_type::real_type;

    explicit mock_openmp_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}

    // make non-virtual functions publicly visible
    using base_type::generate_q;
    using base_type::run_device_kernel;
    using base_type::setup_data_on_device;

    // parameter setter
    void set_cost(const real_type cost) { base_type::cost_ = cost; }
    void set_QA_cost(const real_type QA_cost) { base_type::QA_cost_ = QA_cost; }

    // getter for internal variable
    std::shared_ptr<const std::vector<real_type>> &get_alpha_ptr() { return base_type::alpha_ptr_; }
    const std::vector<std::vector<real_type>> &get_device_data() const { return *base_type::data_ptr_; }
};
