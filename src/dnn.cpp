
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "../lib/dnn.hpp"

double relu(double x) { return x < 0.00 ? 0.00 : x; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double htan(double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }

double relu_dev(double x) { return x < 0.00 ? 0.00 : 1; }
double sigmoid_dev(double x) { return (sigmoid(x) * exp(-x)) / (1 + exp(-x)); }
double htan_dev(double x) { return 1 - pow(htan(x), 2); }

double Node::get_summation() { return summation; }
double Node::get_activation() { return activation; }
double Node::get_error_summation() { return error_summation; }
std::string Node::get_activation_func() { return activation_func; }
std::vector<double> * Node::get_weights() { return &weights; }

void Node::add_error_summation(double val) { error_summation += val; }

void Node::set_summation(double val, bool compute_activation=true) {
    summation = val;
    if(compute_activation) {
        if(activation_func == "sigmoid") activation = sigmoid(summation);
        else if(activation_func == "tanh") activation = htan(summation);
        else activation = relu(summation);
    }
}

void Node::reset() {
    summation = 0.00;
    activation = 0.00;
    error_summation = 0.00;
}

std::vector<double> normalize(std::vector<double> &data) {
    std::vector<double> normalized;
    double max = *(std::max_element(data.begin(), data.end()));
    double min = *(std::min_element(data.begin(), data.end()));
    for(unsigned int i = 0; i < data.size(); i++) normalized.push_back((data[i] - min) / (max - min));

    data.clear();
    return normalized;
}

double mse(std::vector<double> &y, std::vector<double> &yhat) {
    double residual_squared_sum = 0.00;
    for(unsigned int i = 0; i < y.size(); i++) residual_squared_sum += pow(y[i] - yhat[i], 2);
    return residual_squared_sum / y.size();
}

void DeepNet::display(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y) {
    std::cout << "\nSGD-DNN Architecture" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    for(unsigned int l = 0; l < layers.size(); l++) {
        std::cout << "Layer " << l << ": ";
        std::cout << "(" << layers[l].get_shape(0) << ", " << layers[l].get_shape(1) << ") ";
        std::cout << (*layers[l].get_nodes())[0].get_activation_func() << std::endl;
    }
    std::cout << "\nInput:  (" << x.size() << ", " << x[0].size() << ")" << std::endl;
    std::cout << "Output: (" << y.size() << ", " << y[0].size() << ")\n" << std::endl;
}

std::vector<double> DeepNet::predict(std::vector<double> &input, bool normalize_yhat=false) {
    std::vector<double> yhat;
    // fully connected feedforward
    for(unsigned int l = 0; l < layers.size(); l++) {
        std::vector<Node> *nodes; nodes = layers[l].get_nodes();
        for(unsigned int n = 0; n < (*nodes).size(); n++) {
            double dot = 0.00;
            std::vector<double> *weights; weights = (*nodes)[n].get_weights();
            (*nodes)[n].reset();
            for(unsigned int i = 0; i < (*weights).size(); i++) {
                if(l == 0) dot += (*weights)[i] * input[i];
                else dot += (*weights)[i] * (*layers[l-1].get_nodes())[i].get_activation();
            }
            (*nodes)[n].set_summation(dot, true); dot = 0.00; // compute_activation=true
            if(l == layers.size() - 1) yhat.push_back((*nodes)[n].get_activation());
        }
    }

    if(normalize_yhat) yhat = normalize(yhat);
    return yhat;
}

void DeepNet::fit(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y, unsigned int epoch, double learning_rate, double decay_factor) {
    display(x, y);
    // stochastic gradient descent algorithm
    auto start = std::chrono::steady_clock::now();
    for(unsigned int t = 1; t <= epoch; t++) {
        for(unsigned int d = 0; d < x.size(); d++) {
            std::vector<double> yhat = predict(x[d], false);
            // backpropagate
            for(int l = layers.size() - 1; l >= 0; l--) {
                std::vector<Node> *nodes; nodes = layers[l].get_nodes();

                for(unsigned int n = 0; n < (*nodes).size(); n++) {
                    double delta, gradient = 0.00;
                    std::vector<double> *weights; weights = (*nodes)[n].get_weights();
                    // compute delta
                    if(l == layers.size() - 1) delta = (-2.00 / (*nodes).size()) * (y[d][n] - yhat[n]); // output layer (MSE)
                    else delta = (*nodes)[n].get_error_summation();// hidden layers

                    if((*nodes)[n].get_activation_func().compare("sigmoid") == 0) delta *= sigmoid_dev((*nodes)[n].get_summation());
                    else if((*nodes)[n].get_activation_func().compare("tanh") == 0) delta *= htan_dev((*nodes)[n].get_summation());
                    else delta *= relu_dev((*nodes)[n].get_summation());

                    for(unsigned int i = 0; i < (*weights).size(); i++) {
                        // compute gradient
                        if(l != 0) {
                            gradient = delta * (*layers[l-1].get_nodes())[i].get_activation();
                            (*layers[l-1].get_nodes())[i].add_error_summation(delta * (*weights)[i]); // add error summation for hidden layers
                        }
                        else gradient = delta * x[d][i];
                        // update weights
                        (*weights)[i] -= learning_rate * gradient;
                    }
                }
            }
            yhat.clear();
        }
        // calculate cost after optimization
        double cost = 0.00;
        for(unsigned int d = 0; d < x.size(); d++) {
            std::vector<double> yhat = predict(x[d], false);
            cost += mse(y[d], yhat);
            yhat.clear();
        }
        cost /= x.size();
        // display training progress
        unsigned int bar_width = 50;
        unsigned int pos = (int)(bar_width * t / epoch);
        std::cout << "EPOCH #" << t << " [";
        for(unsigned int i = 0; i < bar_width; i++) {
            if(i < pos) std::cout << "|";
            else std::cout << " ";
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "] " << (int)(t * 100 / epoch) << "% (MSE = " << cost << ") Elasped Time = " << elapsed << " sec\r";
        std::cout.flush();

        if(t % (int)(epoch / 10) == 0) learning_rate *= (1 - decay_factor); // step decay adaptive learning rate
    }
    std::cout << "\n";
}

void DeepNet::save(std::string model_name) {
    std::string checkpoint = "./models/" + model_name + "/dnn/checkpoint";
    std::ofstream f(checkpoint, std::ios::trunc);
    if(f.is_open()) {
        for(unsigned int l = 0; l < layers.size(); l++) {
            // save layer shape and activation function
            f << layers[l].get_shape(0) << " " << layers[l].get_shape(1) << " " << (*layers[l].get_nodes())[0].get_activation_func() << " \n";
            std::vector<Node> *nodes; nodes = layers[l].get_nodes();
            // save weights of each node into each line
            for(unsigned int n = 0; n < (*nodes).size(); n++) {
                std::vector<double> *weights; weights = (*nodes)[n].get_weights();
                for(unsigned int i = 0; i < (*weights).size(); i++) f << (*weights)[i] << " ";
                f << "\n";
            }
            f << "/ ";
            if(l != layers.size() - 1) f << "\n";
        }
        layers.clear();
        f.close();
    }
}

bool DeepNet::load(std::string model_name) {
    bool loaded = false;
    std::string checkpoint = "./models/" + model_name + "/dnn/checkpoint";
    std::ifstream f(checkpoint);
    if(f.is_open()) {
        std::string line, val;
        std::vector<unsigned int> architecture;
        bool have_shape, have_activation_func = false;
        unsigned int n = 0, k = 0;
        while(std::getline(f, line)) {
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] != ' ') val += line[i];
                else {
                    if(val.compare("/") == 0) {
                        n = 0; k = 0;
                        architecture.clear();
                        have_shape = false; have_activation_func = false;
                    }
                    else {
                        if(!have_shape) {
                            architecture.push_back(std::stoi(val));
                            if(architecture.size() == 2) have_shape = true;
                        }
                        else if(have_shape && !have_activation_func) {
                            layers.push_back(DeepLayer(architecture[0], architecture[1], val));
                            have_activation_func = true;
                        }
                        else {
                            std::vector<Node> *nodes; nodes = layers[layers.size() - 1].get_nodes();
                            std::vector<double> *weights; weights = (*nodes)[n].get_weights();
                            (*weights)[k] = std::stod(val);
                            k++;

                            if(k == (*weights).size()) { n++; k = 0; }
                        }
                    }
                    val = "";
                }
            }
        }
        loaded = true;
        f.close();
    }
    return loaded;
}


