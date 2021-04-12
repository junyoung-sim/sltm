#!/usr/bin/env python3

""" 
read command line arguments in this following order:
    0      1    2     3         4           5            6           7        8
./run.py mode model symbol date1(start) date2(end) learning_rate iteration backtest
         required         |             optional (only for training)
"""

import os, sys
import matplotlib.pyplot as plt
from datetime import datetime

from lib import *

mode       = sys.argv[1]
model      = sys.argv[2]
symbol     = sys.argv[3]

model_path     = "./models/" + model

def init():
    print("\nModel: {}\nSymbol: {}\n" .format(model, symbol))
    okay = False
    if mode == "train":
        print("Initializing file system...")
        required = [
            "./temp",
            "./data",
            "./models",
            model_path,
            model_path + "/kernels",
            model_path + "/dnn",
            model_path + "/backtest",
            model_path + "/trained-samples",
            model_path + "/res",
            model_path + "/res/npy",
            model_path + "/res/prediction",
            model_path + "/res/confidence"
        ]
        for d in required:
            if not os.path.exists(d): # check if required directory exists; if not, create it
                os.mkdir(d)
        okay = True
    elif mode == "run":
        if os.path.exists(model_path + "/" + model): # checking if trained model exists
            okay = True
        else:
            print("\nRequested model does not exist!\n")
    else:
        print("\nInvalid mode given!\n")
    if okay:
        for root, dirs, files in os.walk("./temp"):
            for f in files:
                os.system("rm -rf ./temp/" + f)
        print("Building encoder...\n")
        os.system("./scripts/build")
    return okay

def train():
    date1         = sys.argv[4]
    date2         = sys.argv[5]
    learning_rate = float(sys.argv[6])
    iteration     = int(sys.argv[7])
    backtest      = float(sys.argv[8])
    # sample and save training dataset from historical data
    dataset = generate_timeseries_dataset(symbol, date1, date2) # ./lib/algo.py 39:56
    # run encoder on training inputs (C coded executable)
    print("\n\nRunning encoder...\n")
    os.system("./encoder " + model)
    print("")
    # read encoded inputs
    encoded = []
    with open("./temp/encoded", "r") as f:
        for line in f.readlines():
            encoded.append([float(val) for val in line.split(" ")])
    dataset["input"] = np.array(encoded)
    print("{} samples (Size = {})\n{}\n" .format(dataset["input"].shape[0], dataset["input"].shape[1], dataset["input"]))    
    # train prediction model
    hyper = {
        "architecture":[[25,25],[25,100],[100,75]],
        "activation": "relu",
        "abs_synapse": 1.0,
        "learning_rate": learning_rate
    }
    print("Prediction DNN = ", hyper, "\n")
    predictor = DeepNeuralNetwork(model_path, hyper)
    predictor.train(dataset, iteration, backtest)
    if not os.path.exists(model_path + "/" + model):
        os.system("touch " + model_path + "/" + model) # create a file indicating that the prediction model is trained

def run():
    data = normalize(mavg(YahooFinance(symbol, "2019-01-01", "yyyy-mm-dd")["prices"][-171:], 50)) # process recent D-121 MAVG 50 input
    with open("./temp/input", "w+") as f: # save input
        for val in data:
            f.write(str(val) + " ")
    # run encoder on input
    print("Running encoder...\n")
    os.system("./encoder " + model)
    # read encoded input
    encoded = []
    with open("./temp/encoded", "r") as f:
        encoded = np.array([[float(val) for val in f.readline().split(" ")]])
    print("\nEncoded input:\n", encoded)
    # run prediction model
    predictor = DeepNeuralNetwork(model_path)
    result = smoothing(predictor.run(encoded)[0])
    print("Model Prediction:\n", np.array(result), "\n")
    # save prediction results
    plt.plot(result, color="red")
    plt.savefig(model_path + "/res/prediction/" + datetime.today().strftime("%Y-%m-%d") + ".png")
    with open(model_path + "/res/npy/" + datetime.today().strftime("%Y-%m-%d") + ".npy", "wb") as f:
        np.save(f, result)
    # run confidence evaluation
    confidence_evaluation(model_path, symbol)

if __name__ == "__main__":
    if init():
        if mode == "train":
            train()
        else:
            run()
