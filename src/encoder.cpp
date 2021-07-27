
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <tuple>
#include "../lib/encoder.hpp"

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
            std::vector<std::vector<float>> *kernel; kernel = layer[l].get_kernel();
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
        f1.close();
    }
}

void Encoder::load() {
    std::string line, val;
    // load encoder parameters
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
                    if(val_count == CONV_SIZE) conv_size = stoul(val);
                    else if(val_count == STRIDE) stride = stoul(val);
                    else if(val_count == PADDING) padding = stoi(val);
                    else if(val_count == POOL_TYPE) pool_type = val;
                    else pool_size = stoul(val);
                    val_count++;
                    val = "";
                }
            }
            // read kernel of each layer
            std::vector<std::vector<float>> kernel;
            f2.open(path + "/kernels/kernel" + std::to_string(layer.size()));
            if(f2.good()) {
                while(getline(f2, line)) {
                    std::vector<float> row;
                    for(unsigned int i = 0; i < line.length(); i++) {
                        if(line[i] != ' ') val += line[i];
                        else {
                            row.push_back(stof(val));
                            val = "";
                        }
                    }
                    kernel.push_back(row);
                }
                f2.close();
            }
            // construct layer
            layer.push_back(Layer(conv_size, stride, padding, pool_type, pool_size, kernel));
        }
        f1.close();
    }
    // display encoder parameters
    std::cout << "Encoder Parameters" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "     conv_size  stride  padding  pool_type  pool_size" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    for(unsigned int l = 0; l < layer.size(); l++) {
        std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *parameters; parameters = layer[l].get_parameters();
        std::cout << "#" << l << ":      " << get<CONV_SIZE>(*parameters) << "        " << get<STRIDE>(*parameters) << "        " << get<PADDING>(*parameters) << "        " << get<POOL_TYPE>(*parameters) << "         " << get<POOL_SIZE>(*parameters) << " " << std::endl;
    }
    // read input
    f1.open("./temp/input");
    if(f1.good()) {
        while(getline(f1, line)) { // each input is written in each line
            std::vector<float> row;
            std::vector<std::vector<float>> input;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    row.push_back(stof(val));
                    if(row.size() == 11) { // reshaping input into 2D image (raster scanning order)
                         input.push_back(row);
                         row.clear();
                    }
                    val = "";
                }
            }
            dataset.push_back(input);
            input.clear();
        }
        f1.close();
    }
}

void Encoder::add_layer(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size) {
    std::vector<std::vector<float>> empty_kernel;
    layer.push_back(Layer(conv_size, stride, padding, pool_type, pool_size, empty_kernel));
}

void Encoder::encode() {
    std::vector<std::vector<float>> *input;
    std::vector<std::vector<float>> pad, convolved, pooled;
    std::vector<std::vector<std::vector<float>>> encoded;
    for(unsigned int d = 0; d < dataset.size(); d++) {
        input = &dataset[d]; // initial input
        for(unsigned int l = 0; l < layer.size(); l++) {
            std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> *parameters; parameters = layer[l].get_parameters();
            // padding
            if(get<PADDING>(*parameters)) {
                for(unsigned int i = 0; i < (*input).size() + 2; i++) {
                    pad.push_back(std::vector<float>((*input).size() + 2, 0.00)); 
                }
                for(unsigned int i = 1; i < pad.size() - 1; i++) {
                    for(unsigned int j = 1; j < pad[i].size() - 1; j++) {
                        pad[i][j] = (*input)[i-1][j-1];
                    }
                }
                input = &pad;
            }
            // convolution
            std::vector<std::vector<float>> *kernel; kernel = layer[l].get_kernel();
            unsigned int conv_size = get<CONV_SIZE>(*parameters);
            unsigned int stride = get<STRIDE>(*parameters);
            for(unsigned int r = 0; r <= (*input).size() - conv_size; r += stride) {
                std::vector<float> row;
                for(unsigned int c = 0; c <= (*input)[r].size() - conv_size; c += stride) {
                    float dot = 0.00;
                    for(unsigned int i = r; i < r + conv_size; i++) {
                        for(unsigned int j = c; j < c + conv_size; j++) {
                            dot += (*input)[i][j] * (*kernel)[i-r][j-c];
                        }
                    }
                    row.push_back(dot < 0.00 ? 0.00 : dot); // threshold matrix multiplication with ReLU
                    dot = 0.00;
                }
                convolved.push_back(row);
            }
            // pooling (max or avg)
            pad.clear(); pooled.clear();
            std::string pool_type = get<POOL_TYPE>(*parameters);
            unsigned int pool_size = get<POOL_SIZE>(*parameters);
            for(unsigned int r = 0; r <= convolved.size() - pool_size; r += pool_size) {
                std::vector<float> row;
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
                    value = 0.00;
                }
                pooled.push_back(row);
            }
            convolved.clear();
            input = &pooled;
        }
        dataset[d].clear();
        encoded.push_back(*input);
    }
    // save encoded inputs
    std::ofstream f("./temp/encoded");
    if(f.is_open()) {
        for(unsigned int d = 0; d < encoded.size(); d++) {
            std::vector<float> data;
            // flatten the encoded input
            for(unsigned int i = 0; i < encoded[d].size(); i++) {
                for(unsigned int j = 0; j < encoded[d][i].size(); j++) {
                    data.push_back(encoded[d][i][j]);
                }
            }
            for(unsigned int i = 0; i < data.size(); i++) { // write each value of the encoded input splitted by spaces
                f << data[i];
                if(i != data.size() - 1) f << " ";
            }
            if(d != encoded.size() - 1) f << "\n";
        }
        f.close();
    }
    encoded.clear();
}
