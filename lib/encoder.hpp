
#ifndef __ENCODER_HPP_
#define __ENCODER_HPP_

#define CONV_SIZE 0
#define STRIDE    1
#define PADDING   2
#define POOL_TYPE 3
#define POOL_SIZE 4

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <tuple>
#include <ctime>

class ConvPool2D
{
private:
    std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> parameters;
    std::vector<std::vector<double>> kernel;
public:
    ConvPool2D(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size, std::vector<std::vector<double>> &kernel_source): kernel(kernel_source) {
        kernel_source.clear();
        parameters = make_tuple(conv_size, stride, padding, pool_type, pool_size);
        if(kernel.empty()) {
            for(unsigned int i = 0; i < conv_size; i++) {
                std::vector<double> row;
                for(unsigned int j = 0; j < conv_size; j++) {
                    double init = -0.5 + (double)rand() / RAND_MAX * (0.5 - (-0.5)); // kernel values: -0.5 <= x <= 0.5
                    row.push_back(init);
                }
                kernel.push_back(row);
                row.clear();
            }
        }
    }
   std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *get_parameters() { return &parameters; }
   std::vector<std::vector<double>> *get_kernel() { return &kernel; }
};

class Encoder
{
private:
    std::string path;
    std::vector<ConvPool2D> layer;
public:
    Encoder(std::string model_name): path("./models/" + model_name + "/encoder") {}
    void add_layer(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size);
    std::vector<std::vector<double>> encode(std::vector<std::vector<std::vector<double>>> &dataset);
    void save();
    void load();
};

#endif
