/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "plssvm/constants.hpp"     // plssvm::kernel_index_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type, plssvm::kernel_function

#include <utility>  // std::forward
#include <vector>   // std::vector

#include <iostream>

#include <mpi.h>

namespace plssvm::openmp {

namespace detail {

template <kernel_type kernel, typename real_type, typename... Args>
void device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, Args &&...args) {
    const auto dept = static_cast<kernel_index_type>(d.size());

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype mpi_real_type;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(real_type), &mpi_real_type);

    MPI_Barrier(MPI_COMM_WORLD);

    // only root rank remembers the right side of the equation

    if (rank != 0) {
        std::fill(ret.begin(), ret.end(), real_type{ 0.0 });
    }

    int comparrisonCount = 0;

    double startTime = MPI_Wtime();

    double n = data.size() - 1; // could also use dept variable
                                // overflow at n > 2,000,000,000 = 2 * 10^9
    double t = world_size;

    // prepare variables for a MNF
    double a = 1;
    double b = -(2 * n + 1);

    double c_lower = n * n + n - (((n * n + 1) * rank) / t);  // is a squared function noticably faster?
    double c_upper = n * n + n - (((n * n + 1) * (rank + 1)) / t);

    // compute the lower and upper bound with a simple MNF formula

    int lowerBound = n - static_cast<int>(std::floor(((-b - sqrt(b * b - 4 * a * c_lower)) / (2 * a))));
    int upperBound = n - static_cast<int>(std::floor(((-b - sqrt(b * b - 4 * a * c_upper)) / (2 * a))));

    double boundTime = MPI_Wtime();

    for (kernel_index_type i = lowerBound; i < upperBound; ++i) {
        real_type ret_i = 0.0;
        for (kernel_index_type j = 0; j <= i; ++j) {
            comparrisonCount++;
            const real_type temp = (kernel_function<kernel>(data[i], data[j], std::forward<Args>(args)...) + QA_cost - q[i] - q[j]) * add;
            if (i == j) {
                ret_i += (temp + cost * add) * d[i];
            } else {
                ret_i += temp * d[j];
                ret[j] += temp * d[i];
            }
        }
        ret[i] += ret_i;
    }

    double compTime = MPI_Wtime();


    if (rank != 0) {
        MPI_Send(&ret[0], ret.size(), mpi_real_type, 0, 1, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        std::vector<real_type> temp_ret(ret.size());
        for (kernel_index_type i = 1; i < world_size; ++i) {
            MPI_Recv(&temp_ret[0], temp_ret.size(), mpi_real_type, i, 1, MPI_COMM_WORLD, &status);
            for (kernel_index_type j = 0; j < temp_ret.size(); ++j) {
                ret[j] += temp_ret[j];
            }
        }
    }
    
    double sendTime = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " ; " << boundTime - startTime << " ; " << compTime - boundTime << " ; " << sendTime - compTime << " ; " << sendTime - startTime << " ; " << comparrisonCount << std::endl;


}

}  // namespace detail

template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add) {
    detail::device_kernel<kernel_type::linear>(q, ret, d, data, QA_cost, cost, add);
}
template void device_kernel_linear(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float);
template void device_kernel_linear(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double);

template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    detail::device_kernel<kernel_type::polynomial>(q, ret, d, data, QA_cost, cost, add, degree, gamma, coef0);
}
template void device_kernel_poly(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const int, const float, const float);
template void device_kernel_poly(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const int, const double, const double);

template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const real_type gamma) {
    detail::device_kernel<kernel_type::rbf>(q, ret, d, data, QA_cost, cost, add, gamma);
}
template void device_kernel_radial(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const float);
template void device_kernel_radial(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const double);

}  // namespace plssvm::openmp
