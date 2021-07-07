#include <plssvm/CSVM.hpp>
#include <plssvm/detail/string_utility.hpp>

namespace plssvm {

template <typename T>
void CSVM<T>::learn() {
    std::vector<real_t> q;
    std::vector<real_t> b = value;
    #pragma omp parallel sections
    {
        #pragma omp section  // generate q
        {
            q = generate_q();
        }
        #pragma omp section  // generate right side from eguation
        {
            b.pop_back();
            b -= value.back();
        }
        #pragma omp section  // generate botom right from A
        {
            QA_cost = kernel_function(data.back(), data.back()) + 1 / cost;
        }
    }

    if (print_info_)
        std::cout << "start CG" << std::endl;
    //solve minimization
    alpha = CG(b, num_features, epsilon, q);
    alpha.emplace_back(-sum(alpha));
    bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}

template <typename T>
real_t CSVM<T>::kernel_function(real_t *xi, real_t *xj, int dim) {  //TODO: kernel as template
    switch (kernel) {
        case kernel_type::linear:
            return mult(xi, xj, dim);
        case kernel_type::polynomial:
            return std::pow(gamma * mult(xi, xj, dim) + coef0, degree);
        case kernel_type::rbf: {
            real_t temp = 0;
            for (int i = 0; i < dim; ++i) {
                temp += (xi[i] - xj[i]);
            }
            return exp(-gamma * temp * temp);
        }
        default:
            throw std::runtime_error("Can not decide wich kernel!");
    }
}

template <typename T>
real_t CSVM<T>::kernel_function(std::vector<real_t> &xi, std::vector<real_t> &xj) {
    switch (kernel) {
        case kernel_type::linear:
            return xi * xj;
        case kernel_type::polynomial:
            return std::pow(gamma * (xi * xj) + coef0, degree);
        case kernel_type::rbf: {
            real_t temp = 0;
            for (int i = 0; i < xi.size(); ++i) {
                temp += (xi - xj) * (xi - xj);
            }
            return exp(-gamma * temp);
        }
        default:
            throw std::runtime_error("Can not decide wich kernel!");
    }
}

template <typename T>
void CSVM<T>::learn(const std::string &input_filename, const std::string &model_filename) {
    // parse data file
    this->parse_file(input_filename);

    //    auto end_parse = std::chrono::high_resolution_clock::now();
    //    if (info) {
    //        std::clog << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << " ms eingelesen" << std::endl
    //                  << std::endl;
    //    }
    loadDataDevice();

    //    auto end_gpu = std::chrono::high_resolution_clock::now();
    //
    //    if (info)
    //        std::clog << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << " ms auf die Gpu geladen" << std::endl
    //                  << std::endl;

    learn();
    //    auto end_learn = std::chrono::high_resolution_clock::now();
    //    if (info)
    //        std::clog << std::endl
    //                  << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << " ms gelernt" << std::endl;

    // write results to model file
    write_model(model_filename);

    //    if (true) {  // TODO: check
    //        std::clog << data.size() << ", " << num_features << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << std::endl;
    //    }
}

// explicitly instantiate template class
template class CSVM<float>;
template class CSVM<double>;

}  // namespace plssvm