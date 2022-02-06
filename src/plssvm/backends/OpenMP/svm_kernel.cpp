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
    //const auto dept = static_cast<kernel_index_type>(d.size());

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
    
    int n = data.size() - 1;
    int t = world_size;

    int lowerBoundI = 0;
    int upperBoundI = 0;
    int lowerBoundJ = 0;
    int upperBoundJ = 0;

    int wh = t / 2;

    if (rank < wh) {
        lowerBoundI = n * (rank) / (wh * 2);
        upperBoundI = n * (rank + 1) / (wh * 2);

        lowerBoundJ = 0;
        upperBoundJ = n / 2;
    } else {
        if (t % 2 == 1) {
            wh++;
        }
        lowerBoundI = n * (rank) / (wh * 2);
        upperBoundI = n * (rank + 1) / (wh * 2);

        lowerBoundJ = 0;
        upperBoundJ = n / 2;
    }

    // compute the lower and upper bound with a simple MNF formula

    //int lowerBound = n - static_cast<int>(std::floor(((-b - sqrt(b * b - 4 * a * c_lower)) / (2 * a))));
    //int upperBound = n - static_cast<int>(std::floor(((-b - sqrt(b * b - 4 * a * c_upper)) / (2 * a))));

    double boundTime = MPI_Wtime();
    
    //std::cout << rank << " ; " << lowerBoundI << " ; " << upperBoundI << " ;; " << lowerBoundJ << " ; " << upperBoundJ << std::endl;

    //#pragma omp parallel for collapse(2) schedule(dynamic)
    for (kernel_index_type i = lowerBoundI; i < upperBoundI; i += OPENMP_BLOCK_SIZE) {
        for (kernel_index_type j = lowerBoundJ; j < upperBoundJ; j += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < upperBoundI; ++ii) {
                real_type ret_iii = 0.0;
                for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < std::min(ii + i + 1, upperBoundJ); ++jj) {
                    //#pragma omp atomic
                    comparrisonCount++;
                    const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                    if (ii + i == jj + j) {
                        ret_iii += (temp + cost * add) * d[ii + i];
                    } else {
                        ret_iii += temp * d[jj + j];
                        //#pragma omp atomic
                        ret[jj + j] += temp * d[ii + i];
                    }
                }
                //#pragma omp atomic
                ret[ii + i] += ret_iii;
            }
        }
    } 

    
    if (rank < wh) {
        if (t % 2 == 1) {
            wh++;
        }

        lowerBoundI = n * (t - rank - 1) / (wh * 2);
        upperBoundI = n * (t - rank) / (wh * 2);

        lowerBoundJ = n / 2;
        upperBoundJ = n;
               
        // std::cout << rank << " ; " << lowerBoundI << " ; " << upperBoundI << " ;; " << lowerBoundJ << " ; " << upperBoundJ << std::endl;
        
        //#pragma omp parallel for collapse(2) schedule(dynamic)
        for (kernel_index_type i = lowerBoundI; i < upperBoundI; i += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type j = lowerBoundJ; j < upperBoundJ; j += OPENMP_BLOCK_SIZE) {
                for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < upperBoundI; ++ii) {
                    real_type ret_iii = 0.0;
                    for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < std::min(ii + i + 1, upperBoundJ); ++jj) {
                        //#pragma omp atomic
                        comparrisonCount++;
                        const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                        if (ii + i == jj + j) {
                            ret_iii += (temp + cost * add) * d[ii + i];
                        } else {
                            ret_iii += temp * d[jj + j];
                            //#pragma omp atomic
                            ret[jj + j] += temp * d[ii + i];
                        }
                    }
                    //#pragma omp atomic
                    ret[ii + i] += ret_iii;
                }
            }
        }      
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

    
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            //std::cout << rank << " ; " << boundTime - startTime << " ; " << compTime - boundTime << " ; " << sendTime - compTime << " ; " << sendTime - startTime << " ; " << comparrisonCount << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
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
