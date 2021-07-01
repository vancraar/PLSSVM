#pragma once
#include "plssvm/CSVM.hpp"
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
#include "plssvm/OpenCL/OpenCL_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
#include "plssvm/CUDA/CUDA_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
#include "plssvm/OpenMP/OpenMP_CSVM.hpp"
#endif
#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockCSVM : public plssvm::CSVM {
  public:
    MockCSVM(real_t cost_, real_t epsilon_, plssvm::kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : plssvm::CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t), (override));
    using plssvm::CSVM::kernel_function;
    // MOCK_METHOD(real_t, kernel_function, (std::vector<real_t> &, std::vector<real_t> &), (override));
    // MOCK_METHOD(real_t, kernel_function, (real_t *, real_t *, int), (override));

    const real_t get_num_data_points() const { return num_data_points; }
    const real_t get_num_features() const { return num_features; }
    std::vector<std::vector<real_t>> get_data() const { return data; }
    const real_t get_gamma() const { return gamma; }
};
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
class MockOpenMP_CSVM : public plssvm::OpenMP_CSVM {
  public:
    MockOpenMP_CSVM(real_t cost_, real_t epsilon_, plssvm::kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : plssvm::OpenMP_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t), (override));

    using plssvm::OpenMP_CSVM::kernel_function;

    // MOCK_METHOD(real_t, kernel_function, (std::vector<real_t> &, std::vector<real_t> &), (override));
    // MOCK_METHOD(real_t, kernel_function, (real_t *, real_t *, int), (override));

    const real_t get_num_data_points() const {
        return num_data_points;
    }
    const real_t get_num_features() const {
        return num_features;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data;
    }
    const real_t get_gamma() const {
        return gamma;
    }
};
#else
#pragma message("Ignore OpenMP backend test")
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
class MockOpenCL_CSVM : public plssvm::OpenCL_CSVM {
  public:
    MockOpenCL_CSVM(real_t cost_, real_t epsilon_, plssvm::kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : plssvm::OpenCL_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t), (override));

    using plssvm::OpenCL_CSVM::kernel_function;

    // MOCK_METHOD(real_t, kernel_function, (std::vector<real_t> &, std::vector<real_t> &), (override));
    // MOCK_METHOD(real_t, kernel_function, (real_t *, real_t *, int), (override));

    const real_t get_num_data_points() const {
        return num_data_points;
    }
    const real_t get_num_features() const {
        return num_features;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data;
    }
    const real_t get_gamma() const {
        return gamma;
    }
};
#else
#pragma message("Ignore OpenCL backend Test")
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
class MockCUDA_CSVM : public plssvm::CUDA_CSVM {
  public:
    MockCUDA_CSVM(real_t cost_, real_t epsilon_, plssvm::kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : plssvm::CUDA_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t), (override));

    using plssvm::CUDA_CSVM::kernel_function;

    // MOCK_METHOD(real_t, kernel_function, (std::vector<real_t> &, std::vector<real_t> &), (override));
    // MOCK_METHOD(real_t, kernel_function, (real_t *, real_t *, int), (override));

    const real_t get_num_data_points() const {
        return num_data_points;
    }
    const real_t get_num_features() const {
        return num_features;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data;
    }
    const real_t get_gamma() const {
        return gamma;
    }
};
#else
#pragma message("Ignore CUDA backend Test")
#endif
