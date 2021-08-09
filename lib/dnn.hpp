
#ifndef __DEEP_NEURAL_NETWORK_HPP_
#define __DEEP_NEURAL_NETWORK_HPP_

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

double relu(double x);
double sigmoid(double x);
double htan(double x);

double relu_dev(double x);
double sigmoid_dev(double x);
double htan_dev(double x);

std::vector<double> normalize(std::vector<double> &data);
double mse(std::vector<double> &y, std::vector<double> &yhat);

class Node
{
private:
    double summation;
    double activation;
    double error_summation;
    std::string activation_func;
    std::vector<double> weights;
public:
    Node() {}
    Node(unsigned int num_of_inputs, std::string _activation_func): activation_func(_activation_func) {
        reset();
        // initialize weight vector with "i" number of random float values (-1.0 ~ 1.0)
        for(unsigned int i = 0; i < num_of_inputs; i++) {
            double init = -1.0 + (double)rand() / RAND_MAX * (1.0 - (-1.0));
            weights.push_back(init);
        }
    }

    double get_summation();
    double get_activation();
    double get_error_summation();
    std::string get_activation_func();
    std::vector<double> *get_weights();

    void add_error_summation(double val);
    void set_summation(double val, bool compute_activation);

    void reset();
};

class DeepLayer
{
private:
    std::vector<unsigned int> shape;
    std::vector<Node> nodes;
public:
    DeepLayer(unsigned int num_of_inputs, unsigned int num_of_nodes, std::string activation_func) {
        shape.push_back(num_of_inputs);
        shape.push_back(num_of_nodes);
        // create "n" number of nodes with "i" number of weights
        for(unsigned int n = 0; n < shape[1]; n++) {
            nodes.push_back(Node(shape[0], activation_func));
        }
    }
    unsigned int get_shape(unsigned int d) { return shape[d]; }
    std::vector<Node> *get_nodes() { return &nodes; }
};

class DeepNet
{
private:
    std::vector<DeepLayer> layers;
    //std::string cost_func;
public:
    DeepNet() {}
    void init(std::vector<std::vector<unsigned int>> architecture, std::string activation_func) {
        /*
            architecture: vector of {num_of_inputs, num_of_outputs}; pass argument as {{}...{}}
            activation_func: "relu", "sigmoid", or "tanh"
            cost_func: "mse" (default), // more to be added
        */
        for(unsigned int l = 0; l < architecture.size(); l++) {
            layers.push_back(DeepLayer(architecture[l][0], architecture[l][1], activation_func));
        }
    }
    void display(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y);
    std::vector<double> predict(std::vector<double> &input, bool normalize_yhat);
    void fit(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y, unsigned int epoch, double learning_rate, double decay_factor);
    void save(std::string model_name);
    bool load(std::string model_name);
};

#endif
