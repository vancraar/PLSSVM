#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <omp.h> 
#include <sstream> 
#include <cstdlib>
#include <stdlib.h>
#include <utility>
#include <tuple>
#include <math.h>

#include "operators.hpp"
// #include "typedef.hpp"
#include <tuple>


#ifdef WITH_OPENCL
#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include "distribution.hpp"
#include <stdexcept>
#endif

const bool times = 0;



// static const unsigned CUDABLOCK_SIZE = 7;
// static const unsigned BLOCKING_SIZE_THREAD = 2;

class CSVM
{
    public:
        CSVM(real_t, real_t, unsigned, real_t, real_t, real_t, bool);
		void learn(std::string&, std::string&);
        
		const real_t& getB() const { return bias; };
        void load_w();
        std::vector<real_t> predict(real_t*, int, int);
    protected:

    private:
        const bool info; 
        real_t cost;
        const real_t epsilon;
        const unsigned kernel;
        const real_t degree;
        real_t gamma;
        const real_t coef0;
        real_t bias;
		real_t QA_cost;
        std::vector<std::vector<real_t> > data;
		size_t Nfeatures_data;
		size_t Ndatas_data;
        std::vector<real_t> value;
        std::vector<real_t> alpha;


        void learn();
		
        inline real_t kernel_function(std::vector<real_t>&, std::vector<real_t>&);
        inline real_t kernel_function(real_t*, real_t*, int);

        void libsvmParser(std::string&);
        void arffParser(std::string&);
        void writeModel(std::string&);

        void loadDataDevice();

        

		std::vector<real_t> CG(const std::vector<real_t> &b, const int , const real_t );
        
        inline std::vector<real_t> transform_data(const int start_line, const int boundary){
            std::vector<real_t> vec(Nfeatures_data * (Ndatas_data - 1 + boundary));
            #pragma omp parallel for collapse(2)
            for(size_t col = 0; col < Nfeatures_data; ++col){
            for(size_t row = 0; row < Ndatas_data - 1; ++row){
                vec[col * (Ndatas_data - 1 + boundary) + row ] = data[row][col];
		}
	}
	return vec;

}
        inline void loadDataDevice(const int device, const int boundary, const int start_line, const int number_lines, const std::vector<real_t> data);
        #ifdef WITH_OPENCL
            inline void resizeData(int boundary);
            inline void resizeData(const int device, int boundary);
            inline void resizeDatalast(int boundary);
            inline void resizeDatalast(const int device, int boundary);
            // inline void resize(const int old_boundary,const int new_boundary);
            distribution distr;
	        opencl::manager_t manager{"../platform_configuration.cfg"};
	        opencl::device_t first_device;
            std::vector<cl_kernel> kernel_q_cl;
            std::vector<cl_kernel> svm_kernel_linear;
            std::vector<opencl::DevicePtrOpenCL<real_t> > datlast_cl;
            std::vector<opencl::DevicePtrOpenCL<real_t> > data_cl;
        #endif

        #ifdef WITH_CUDA
            std::vector<real_t *> data_d;
            std::vector<real_t *> datlast_d;
            real_t *w_d;
        #endif
    };



