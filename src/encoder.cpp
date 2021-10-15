
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include "../lib/encoder.hpp"

void Encoder::add_layer(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size) {
    std::vector<std::vector<double>> empty_kernel;
    layer.push_back(ConvPool2D(conv_size, stride, padding, pool_type, pool_size, empty_kernel));
}

std::vector<std::vector<double>> Encoder::encode(std::vector<std::vector<std::vector<double>>> &dataset) {
    std::vector<std::vector<double>> *input;
    std::vector<std::vector<double>> pad, convolved, pooled;
    std::vector<std::vector<double>> encoded;
    for(unsigned int d = 0; d < dataset.size(); d++) {
        input = &dataset[d]; // initial data
        for(unsigned int l = 0; l < layer.size(); l++) {
            std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *parameters; parameters = layer[l].get_parameters();
            // padding
            if(get<PADDING>(*parameters)) {
                for(unsigned int i = 0; i < (*input).size() + 2; i++) {
                    pad.push_back(std::vector<double>((*input).size() + 2, 0.00)); 
                }
                for(unsigned int i = 1; i < pad.size() - 1; i++) {
                    for(unsigned int j = 1; j < pad[i].size() - 1; j++) {
                        pad[i][j] = (*input)[i-1][j-1];
                    }
                }
                input = &pad;
            }
            // convolution
            std::vector<std::vector<double>> *kernel; kernel = layer[l].get_kernel();
            unsigned int conv_size = get<CONV_SIZE>(*parameters); 
            unsigned int stride = get<STRIDE>(*parameters);
            for(unsigned int r = 0; r <= (*input).size() - conv_size; r += stride) {
                std::vector<double> row;
                for(unsigned int c = 0; c <= (*input)[r].size() - conv_size; c += stride) {
                    float dot = 0.00;
                    for(unsigned int i = r; i < r + conv_size; i++) {
                        for(unsigned int j = c; j < c + conv_size; j++) {
                            dot += (*input)[i][j] * (*kernel)[i-r][j-c];
                        }
                    }
                    row.push_back(dot < 0.00 ? 0.00 : dot); // ReLU
                }
                convolved.push_back(row);
                row.clear();
            }
            // pooling (max or avg)
            pad.clear(); pooled.clear();
            std::string pool_type = get<POOL_TYPE>(*parameters);
            unsigned int pool_size = get<POOL_SIZE>(*parameters);
            for(unsigned int r = 0; r <= convolved.size() - pool_size; r += pool_size) {
                std::vector<double> row;
                for(unsigned int c = 0; c <= convolved[r].size() - pool_size; c += pool_size) {
                    float value = 0.00;
                    for(unsigned int i = r; i < r + pool_size; i++) {
                        for(unsigned int j = c; j < c + pool_size; j++) {
                            if(pool_type == "max") value < convolved[i][j] ? value = convolved[i][j] : value = value;
                            else if(pool_type == "avg") value += convolved[i][j];
                            else {}
                        }
                    }
                    if(pool_type == "avg") value /= pool_size * pool_size;
                    row.push_back(value);
                }
                pooled.push_back(row);
                row.clear();
            }
            convolved.clear();
            input = &pooled;
        }
        // flatten encoded feature map of input
        std::vector<double> flatten;
        for(unsigned int i = 0; i < (*input).size(); i++) {
            for(unsigned int j = 0; j < (*input)[i].size(); j++) {
                flatten.push_back((*input)[i][j]);
            }
        }
        encoded.push_back(flatten);
        pooled.clear(); flatten.clear();
    }

    dataset.clear();
    return encoded;
}

void Encoder::save() {
    std::ofstream f1(path + "/layers");
    if(f1.is_open()) {
        for(unsigned int l = 0; l < layer.size(); l++) {
            // save parameters of each layer into each line
            std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *parameters; parameters = layer[l].get_parameters();
            f1 << get<CONV_SIZE>(*parameters) << " ";
            f1 << get<STRIDE>(*parameters) << " ";
            f1 << get<PADDING>(*parameters) << " ";
            f1 << get<POOL_TYPE>(*parameters) << " ";
            f1 << get<POOL_SIZE>(*parameters) << " ";
            if(l != layer.size() - 1) f1 << "\n";
            // save kernel of each layer
            std::vector<std::vector<double>> *kernel; kernel = layer[l].get_kernel();
            std::ofstream f2(path + "/kernels/kernel" + std::to_string(l));
            if(f2.is_open()) {
                // save each row of the kernel into each line
                for(unsigned int i = 0; i < (*kernel).size(); i++) {
                    for(unsigned int j = 0; j < (*kernel)[i].size(); j++) {
                        f2 << (*kernel)[i][j] << " ";
                    }
                    if(i != (*kernel).size() - 1) f2 << "\n";
                }
                f2.close();
            }
        }
        layer.clear();
        f1.close();
    }
}

void Encoder::load() {
    // load encoder parameters
    std::string line, val;
    std::ifstream f1, f2;
    f1.open(path + "/layers");
    if(f1.good()) {
        layer.clear();
        bool padding; std::string pool_type; unsigned int conv_size, pool_size, stride;
        while(getline(f1, line)) {
            // read parameter values of each layer
            unsigned int val_count = 0;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    if(val_count == CONV_SIZE) conv_size = std::stoul(val);
                    else if(val_count == STRIDE) stride = std::stoul(val);
                    else if(val_count == PADDING) padding = std::stoi(val);
                    else if(val_count == POOL_TYPE) pool_type = val;
                    else pool_size = std::stoul(val);
                    val_count++;
                    val = "";
                }
            }
            // read kernel of each layer
            std::vector<std::vector<double>> kernel;
            f2.open(path + "/kernels/kernel" + std::to_string(layer.size()));
            if(f2.good()) {
                while(getline(f2, line)) {
                    std::vector<double> row;
                    for(unsigned int i = 0; i < line.length(); i++) {
                        if(line[i] != ' ') val += line[i];
                        else {
                            row.push_back(std::stod(val));
                            val = "";
                        }
                    }
                    kernel.push_back(row);
                    row.clear();
                }
                f2.close();
            }
            // construct layer
            layer.push_back(ConvPool2D(conv_size, stride, padding, pool_type, pool_size, kernel));
        }
        f1.close();
    }
    // display encoder parameters
    std::cout << "Encoder Parameters" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "     conv_size  stride  padding  pool_type  pool_size  kernel" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    for(unsigned int l = 0; l < layer.size(); l++) {
        std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *parameters; parameters = layer[l].get_parameters();
        std::cout << "#" << l << ":      " << get<CONV_SIZE>(*parameters) << "        " << get<STRIDE>(*parameters) << "        " << get<PADDING>(*parameters) << "        " << get<POOL_TYPE>(*parameters) << "         " << get<POOL_SIZE>(*parameters) << "     ";

        std::vector<std::vector<double>> *kernel; kernel = layer[l].get_kernel();
        for(unsigned int i = 0; i < (*kernel).size(); i++) {
            std::cout << "[";
            for(unsigned int j = 0; j < (*kernel)[i].size(); j++) {
                std::cout << (*kernel)[i][j];
                if(j != (*kernel)[i].size() - 1) std::cout << " ";
            }
            std::cout << "]";
            if(i != (*kernel).size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }

}

