
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <tuple>
#include "../lib/futures.hpp"

using namespace std;

void Futures::save() {
    ofstream f1(path + "/layers");
    if(f1.is_open()) {
        for(unsigned int l = 0; l < layer.size(); l++) {
            vector<vector<float>> kernel = layer[l].get_kernel();
            // save the attributes of each layer 
            f1 << get<CONV_SIZE>(layer[l].get_attributes()) << " ";
            f1 << get<STRIDE>(layer[l].get_attributes()) << " ";
            f1 << get<PADDING>(layer[l].get_attributes()) << " ";
            f1 << get<POOL_TYPE>(layer[l].get_attributes()) << " ";
            f1 << get<POOL_SIZE>(layer[l].get_attributes()) << " ";
            if(l != layer.size() - 1) f1 << "\n";
            // save each layer's kernel
            ofstream f2(path + "/kernels/kernel" + to_string(l));
            if(f2.is_open()) {
                for(unsigned int i = 0; i < kernel.size(); i++) {
                    for(unsigned int j = 0; j < kernel[i].size(); j++) {
                        f2 << kernel[i][j] << " ";
                    }
                    if(i != kernel.size() - 1) f2 << "\n";
                }
                f2.close();
            }
        }
        f1.close();
    }
}

void Futures::load() {
    string line, val;
    unsigned int amount_of_layers = 0;
    vector<Layer> layers_read;

    ifstream f1, f2;
    f1.open(path + "/layers");
    if(f1.good()) {
        bool padding;
        string pool_type;
        unsigned int conv_size, pool_size, stride;
        while(getline(f1, line)) {
            amount_of_layers++;
            unsigned int val_count = 0;
            for(unsigned int i = 0; i < line.length(); i++) {
                // sort out the values of the attributes of each layer
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
            // read each layer's kernel
            vector<vector<float>> kernel;
            f2.open(path + "/kernels/kernel" + to_string(amount_of_layers - 1));
            if(f2.good()) {
                while(getline(f2, line)) {
                    vector<float> row;
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
            layers_read.push_back(Layer(conv_size, stride, padding, pool_type, pool_size, kernel));
        }
        f1.close();
    }
    if(!layers_read.empty()) layer = layers_read;
    // display layer attributes
    cout << "Encoder Attributes" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "     conv_size  stride  padding  pool_type  pool_size" << endl;
    cout << "-------------------------------------------------------" << endl;
    for(unsigned int l = 0; l < layer.size(); l++) {
        cout << "#" << l << ":      " << get<CONV_SIZE>(layer[l].get_attributes()) << "        " << get<STRIDE>(layer[l].get_attributes()) << "        " << get<PADDING>(layer[l].get_attributes()) << "        " << get<POOL_TYPE>(layer[l].get_attributes()) << "         " << get<POOL_SIZE>(layer[l].get_attributes()) << " " << endl;
    }
    // load input from ./temp (and output, if applicable)
    f1.open("./temp/input");
    if(f1.good()) {
        cout << "\nReading ./temp/input ... ";
        while(getline(f1, line)) {
            vector<float> row;
            vector<vector<float>> input;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    row.push_back(stof(val));
                    if(row.size() == 11) { // *** HARD-CODED-PARAMETER; SET ACCORDINGLY WITH PARAMETER IN "./lib/modules.py:30" ***
                         input.push_back(row);
                         row.clear();
                    }
                    val = "";
                }
            }
            dataset.push_back(Data());
            dataset[dataset.size() - 1].set_input(input);
        }
        cout << "DONE!" << endl;
        f1.close();
    }
    f1.open("./temp/output");
    if(f1.good()) {
        cout << "Reading ./temp/output ... ";
        for(unsigned int d = 0; d < dataset.size(); d++) {
            getline(f1, line);
            vector<float> output;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    output.push_back(stof(val));
                    val = "";
                }
            }
            dataset[d].set_output(output);
        }
        cout << "DONE!" << endl;
        f1.close();
    }
}

void Futures::add_layer(unsigned int conv_size, unsigned int stride, bool padding, string pool_type, unsigned int pool_size) {
    vector<vector<float>> empty_kernel;
    layer.push_back(Layer(conv_size, stride, padding, pool_type, pool_size, empty_kernel));
}

