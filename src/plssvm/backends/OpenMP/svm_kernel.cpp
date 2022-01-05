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

#include <fstream>
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

    /*
    if (rank == 0) {
        std::cout << "d: " << std::endl;
        for (int i = 0; i < d.size(); i++) {
            std::cout << i << ": " << d[i] << std::endl;
        }
    
        std::cout << "ret: " << std::endl;
        for (int i = 0; i < ret.size(); i++) {
            std::cout << i << ": " << ret[i] << std::endl;
        }
    
        std::cout << "cost: " << cost << std::endl;
        std::cout << "add: " << add << std::endl;
        std::cout << "QA_cost: " << QA_cost << std::endl;
    
        std::cout << "data: " << std::endl;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
    */

    if (rank != 0) {
        std::fill(ret.begin(), ret.end(), real_type{ 0.0 });
    }


    //std::cout << data.size() << " ; " << data[0].size() << " ; " << (rank + 1) * data.size() / world_size << std::endl;


    for (kernel_index_type j = 0; j < dept; j += OPENMP_BLOCK_SIZE) {
        for (kernel_index_type i = j; i < dept; i += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < dept; ++ii) {
                real_type ret_iii = 0.0;
                for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < dept; ++jj) {
                    if (ii + i < (rank+1)*data.size()/world_size) {
                        if (ii + i >= jj + j) {
                            const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                            /*
                            std::cout << "data[" << ii + i << "]:" << std::endl;
                            for (int a = 0; a < data[ii+i].size(); a++) {
                                std::cout << a << ": " << data[ii + i][a] << std::endl;
                            }
                            std::cout << "data[" << jj + j << "]:" << std::endl;
                            for (int a = 0; a < data[jj + j].size(); a++) {
                                std::cout << a << ": " << data[jj + j][a] << std::endl;
                            }
                            */
                            
                            //std::cout << "Kernel: " << kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) << std::endl;
                            //std::cout << "temp: " << temp << " ; " << ii+i << " ; " << jj+j << std::endl;
                            
                            if (ii + i == jj + j) {
                                ret_iii += (temp + cost * add) * d[ii + i];
                            } else {
                                ret_iii += temp * d[jj + j];
                                /*
                                if (jj + j == 0) {
                                    std::cout << temp << " ; " << d[ii + i] << std::endl;
                                    std::cout << temp * d[ii + i] << std::endl;
                                }*/
                                ret[jj + j] += temp * d[ii + i];
                            }
                        }
                    }
                }
                /*
                if (ii + i == 0) {
                    std::cout << ret_iii << std::endl;
                }*/
                ret[ii + i] += ret_iii;
            }
        }
    }

    /*
    for (int i = 0; i < world_size; i++) {
        if (rank == i) {
            std::cout << "ret " << rank << ":" << std::endl;
            for (int j = 0; j < ret.size(); j++) {
                std::cout << ret[j] << "; ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */
    double startTime = MPI_Wtime();

    if (rank != 0) {
        MPI_Send(&ret[0], ret.size(), mpi_real_type, 0, 1, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        std::vector<real_type> temp_ret(ret.size());
        int received = 0;
        for(int i = 1; i < world_size; i++){ 
            MPI_Recv(&temp_ret[0], temp_ret.size(), mpi_real_type, i, 1, MPI_COMM_WORLD, &status);
            for (int j = 0; j < temp_ret.size(); j++) {
                ret[j] += temp_ret[j];
            }
            received++;
        }
    }
    
    MPI_Bcast(&ret[0], ret.size(), mpi_real_type, 0, MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    std::cout << "time: " << endTime - startTime << std::endl;

    /*
    if (rank == 0) {
        std::cout << "ret: " << std::endl;
        for (int i = 0; i < ret.size(); i++) {
            std::cout << i << ": " << ret[i] << std::endl;
        }
    }*/


    /*
    std::ofstream myfile;
    myfile.open("data.txt", std::ios_base::app);
    myfile << "\nret: " << std::endl;*/
    // myfile.close();
 
    
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
