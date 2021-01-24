
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <tuple>

#define CONV_SIZE 0
#define STRIDE    1
#define PADDING   2
#define POOL_TYPE 3
#define POOL_SIZE 4

class Data
{
private:
   std::vector<std::vector<float>> input;
   std::vector<float> output;
public:
   Data() {}
   void set_input(std::vector<std::vector<float>> source) { input = source; }
   void set_output(std::vector<float> source) { output = source; }
   std::vector<std::vector<float>> get_input() { return input; }
   std::vector<float> get_output() { return output; }
};

class Layer
{
private:
    std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> attributes;
    std::vector<std::vector<float>> kernel;
public:
    Layer(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size, std::vector<std::vector<float>> kernel_source): kernel(kernel_source) {
        // layer configurations
        attributes = make_tuple(conv_size, stride, padding, pool_type, pool_size);
        if(kernel.empty()) {
            // if no kernel is given, then generate a random convolution kernel
            for(unsigned int i = 0; i < conv_size; i++) {
                std::vector<float> row;
                for(unsigned int j = 0; j < conv_size; j++) {
                    float val = -0.5 + (float)rand() / RAND_MAX * (0.5 - (-0.5)); // kernel values: -0.5 <= x <= 0.5
                    row.push_back(val);
                }
                kernel.push_back(row);
            }
        }
    }
    std::tuple<unsigned int, unsigned int, bool, std::string, unsigned int> get_attributes() { return attributes; }
    std::vector<std::vector<float>> get_kernel() { return kernel; }
};

class Futures
{
private:
    std::string model, path;
    std::vector<Layer> layer;
    std::vector<Data> dataset;
public:
    Futures(std::string name): model(name), path("./models/" + name) {}
    void save();
    void load();
    void add_layer(unsigned int conv_size, unsigned int stride, bool padding, std::string pool_type, unsigned int pool_size);
    void encode();
    void train(float learning_rate, unsigned int iteration, bool backtest);
    void run();
};

