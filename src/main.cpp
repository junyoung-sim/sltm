
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <ctime>
#include "../lib/encoder.hpp"
#include "../lib/dnn.hpp"

std::vector<std::vector<double>> read(std::string path) {
    std::ifstream f(path);
    std::vector<std::vector<double>> dataset;
    if(f.is_open()) {
        std::string line, val;
        while(getline(f, line)) {
            std::vector<double> data;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    data.push_back(std::stod(val));
                    val = "";
                }
            }
            dataset.push_back(data);
            data.clear();
        }
        f.close();
    }
    return dataset;
}

std::vector<std::vector<std::vector<double>>> reshape(std::vector<std::vector<double>> &raw_dataset, unsigned int reshape_size) {
    // reshaping vector into symmetrical matrix
    std::vector<std::vector<std::vector<double>>> reshaped_dataset;
    for(unsigned int d = 0; d < raw_dataset.size(); d++) {
        std::vector<double> row;
        std::vector<std::vector<double>> reshaped;
        for(unsigned int i = 0; i < raw_dataset[d].size(); i++) {
            row.push_back(raw_dataset[d][i]);
            if(row.size() == reshape_size) {
                reshaped.push_back(row);
                row.clear();
            }
        }
        reshaped_dataset.push_back(reshaped);
        reshaped.clear();
    }
    raw_dataset.clear();
    return reshaped_dataset;
}

bool init(std::string mode, std::string model_name) {
    bool okay = false;
    std::cout << "\nInitializing SLTM...\n\n";
    if(mode.compare("train") == 0) {
        // list of required directories
        std::vector<std::string> required = {
            "./models/" + model_name,
            "./models/" + model_name + "/dnn",
            "./models/" + model_name + "/encoder",
            "./models/" + model_name + "/encoder/kernels",
            "./models/" + model_name + "/backtest",
            "./models/" + model_name + "/res"
        };
        // check if required dircetories exists, create them if not
        for(unsigned int i = 0; i < required.size(); i++) {
            std::filesystem::directory_entry entry { required[i] };
            if(!entry.exists()) {
                std::string cmd = "mkdir " + required[i];
                std::system(cmd.c_str());
            }
        }
        okay = true;
        required.clear();
    }
    else if(mode.compare("run") == 0) {
        // check if trained model checkpoint exists
        std::string checkpoint = "./models/" + model_name + "/dnn/checkpoint";
        std::ifstream f(checkpoint);
        if(f.is_open()) {
            okay = true;
            f.close();
        }
        else std::cout << "No trained checkpoint for model named <" << model_name << "> exists!" << std::endl;
    }
    else std::cout << "Invalid mode given!" << std::endl;

    if(okay) std::system("rm -rf ./temp && mkdir temp");
    return okay;
}

void train(std::string symbol, std::string start, std::string end, unsigned int epoch, double learning_rate, double decay_factor, double backtest) {
    // sample moving average time series dataset
    std::string cmd = "./scripts/data.py sample_timeseries_dataset " + symbol + " " + start + " " + end;
    std::system(cmd.c_str());
    // read dataset
    std::vector<std::vector<double>> raw_input = read("./temp/input");
    std::vector<std::vector<std::vector<double>>> reshaped_input = reshape(raw_input, 11); // raw input destroyed after reshaping
    std::vector<std::vector<double>> output = read("./temp/output");

    // encode reshaped input (ConvPool2D)
    Encoder encoder(symbol);
    encoder.add_layer(2, 1, false, "max", 2);
    encoder.load(); // overwrites encoder with the existing encoder parameters
    std::vector<std::vector<double>> encoded = encoder.encode(reshaped_input);
    encoder.save();

    // partition datset
    std::vector<std::vector<double>> encoded_train, encoded_test;
    std::vector<std::vector<double>> output_train, output_test;
    for(unsigned int d = 0; d < encoded.size(); d++) {
        if(d < encoded.size() - (int)(encoded.size() * backtest)) {
            encoded_train.push_back(encoded[d]);
            output_train.push_back(output[d]);
        }
        else {
            encoded_test.push_back(encoded[d]);
            output_test.push_back(output[d]);
        }
    }
    encoded.clear(); output.clear();

    // train deep neural network
    DeepNet model;
    if(!model.load(symbol)) model.init({{25,50},{50,50}}, "relu");
    model.fit(encoded_train, output_train, epoch, learning_rate, decay_factor);

    // backtest
    if(backtest != 0.0) {
        std::ofstream f1("./models/" + symbol + "/backtest/actual");
        std::ofstream f2("./models/" + symbol + "/backtest/backtest");
        if(f1.is_open() && f2.is_open()) {
            double backtest_cost = 0.00;
            for(unsigned int d = 0; d < encoded_test.size(); d++) {
                std::vector<double> yhat = model.predict(encoded_test[d], true);
                backtest_cost += mse(output_test[d], yhat);
                if(f1.is_open() && f2.is_open()) {
                    for(unsigned int i = 0; i < output_test[d].size(); i++) {
                        f1 << output_test[d][i]; f2 << yhat[i];
                        if(i != output_test[d].size() - 1) { f1 << " "; f2 << " "; }
                    }
                    if(d != encoded_test.size() - 1) { f1 << "\n"; f2 << "\n"; }
                }
                yhat.clear();
            }
            backtest_cost /= encoded_test.size();
            std::cout << "\nBacktesting MSE = " << backtest_cost << std::endl;
            f1.close(); f2.close();
        }
        // plot
        cmd = "./scripts/plot_backtest.py " + symbol;
        std::system(cmd.c_str());
    }

    model.save(symbol);
}

void run(std::string symbol) {
    // download recent moving average time series
    std::string cmd = "./scripts/data.py sample_recent_input " + symbol;
    std::system(cmd.c_str());
    // read data
    std::vector<std::vector<double>> raw_input = read("./temp/input");
    std::vector<std::vector<std::vector<double>>> reshaped_input = reshape(raw_input, 11); // raw input destroyed after reshaping

    // encode reshaped input (ConvPool2D)
    Encoder encoder(symbol);
    encoder.add_layer(2, 1, false, "max", 2);
    encoder.load(); // overwrites encoder with the existing encoder parameters
    std::vector<std::vector<double>> encoded = encoder.encode(reshaped_input);
    encoder.save();

    // load deep neural network
    DeepNet model;
    model.load(symbol);

    // predict
    std::vector<double> yhat = model.predict(encoded[0], true);
    std::ofstream f("./models/" + symbol + "/res/pred");
    if(f.is_open()) {
        for(unsigned int i = 0; i < yhat.size(); i++) {
            f << yhat[i];
            if(i != yhat.size() - 1) f << " ";
        }
        yhat.clear();
        f.close();
    }

    // plot
    cmd = "./scripts/plot.py " + symbol;
    std::system(cmd.c_str());
}

int main(int argc, char *argv[])
{
    std::string mode = argv[1];
    std::string symbol = argv[2]; // also used as model name for convenience

    if(init(mode, symbol)) {
        if(mode == "train") {
            std::string start = argv[3]; std::string end = argv[4];
            unsigned int epoch = std::stoi(argv[5]);
            double learning_rate = std::stod(argv[6]);
            double decay_factor = std::stod(argv[7]);
            double backtest = std::stod(argv[8]);

            train(symbol, start, end, epoch, learning_rate, decay_factor, backtest);
        }
        else run(symbol);
    }

    return 0;
}

