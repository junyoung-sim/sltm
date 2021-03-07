#!/usr/bin/env python3

""" 
read command line arguments in this following order:
    0      1    2     3         4           5            6           7        8
./run.py mode model symbol date1(start) date2(end) learning_rate iteration backtest
"""

import os, sys
import matplotlib.pyplot as plt
from datetime import datetime

from lib import *

mode       = sys.argv[1]
model      = sys.argv[2]
symbol     = sys.argv[3]
model_path = "./models/" + model

def init():
    print("\nModel: {}\nSymbol: {}\n" .format(model, symbol))
    okay = False
    if mode == "train":
        # initialize required root and model file system
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
            model_path + "/res/validation"
        ]
        for d in required:
            if os.path.exists(d) != True:
                os.mkdir(d)
        okay = True
    elif mode == "run":
        if os.path.exists(model_path + "/" + model): # check if identification file exists (i.e., a trained model exists)
            okay = True
        else:
            print("\nRequested model does not exist!\n")
    else:
        print("\nInvalid mode given!\n")
    if okay == True:
        # clear ./temp
        for root, dirs, files in os.walk("./temp"):
            for f in files:
                os.system("rm -rf ./temp/" + f)
        # build encoder
        print("Building encoder...\n")
        os.system("./scripts/build")
    return okay

def train():
    date1         = sys.argv[4]
    date2         = sys.argv[5]
    learning_rate = float(sys.argv[6])
    iteration     = int(sys.argv[7])
    backtest      = float(sys.argv[8])
    # download, process, and save financial time series dataset in ./temp
    dataset = generate_timeseries_dataset(symbol, date1, date2)
    # run the encoder (C coded executable)
    print("\n\nRunning encoder...\n")
    os.system("./encoder " + model)
    print("")
    # read the encoded dataset written in ./temp by the encoder
    encoded = []
    with open("./temp/encoded", "r") as f: # each encoded sample is written in a single line
        for line in f.readlines():
            encoded.append([float(val) for val in line.split(" ")])
    dataset["input"] = np.array(encoded)
    print("{} samples (Size = {})\n{}\n" .format(dataset["input"].shape[0], dataset["input"].shape[1], dataset["input"]))    
    # train deep neural network
    hyper = {
        "architecture":[[25,25],[25,100],[100,75]], # *** HARD-CODED PARAMETER ***
        "activation": "relu",
        "abs_synapse": 1.0,
        "learning_rate": learning_rate
    }
    print("DNN hyperparameters = ", hyper, "\n")
    dnn = DeepNeuralNetwork(model_path, hyper)
    dnn.train(dataset, iteration, backtest) # train neural network and saves trained/backtested plots
    # identification file to verify the model is trained
    os.system("touch " + model_path + "/" + model)

def run():
    # download and process lastest input sample
    data = normalize(mavg(YahooFinance(symbol, "2019-01-01", "yyyy-mm-dd")["prices"][-171:], 50)) # *** HARD-CODED PARAMETER ***
    # write processed input to ./temp
    with open("./temp/input", "w+") as f:
        for val in data:
            f.write(str(val) + " ")
    # run encoder (C coded executable)
    print("Running encoder...\n")
    os.system("./encoder " + model)
    # read encdoed input
    encoded = []
    with open("./temp/encoded", "r") as f:
        encoded = np.array([[float(val) for val in f.readline().split(" ")]])
    print("\nEncoded input:\n", encoded)
    # load model
    dnn = DeepNeuralNetwork(model_path)
    result = smoothing(dnn.run(encoded)[0]) # get result and smooth it w/ Savitzky-Golay filter
    print("Model Prediction:\n", np.array(result), "\n")
    # plot and save result
    plt.plot(result, color="red")
    plt.savefig(model_path + "/res/prediction/" + datetime.today().strftime("%Y-%m-%d") + ".png")
    with open(model_path + "/res/npy/" + datetime.today().strftime("%Y-%m-%d") + ".npy", "wb") as f:
        np.save(f, result)
    # validate trend models by plotting actual price changes on prediction plot
    validate_trend_models(model_path, symbol)

if __name__ == "__main__":
    if init():
        if mode == "train":
            train()
        elif mode == "run":
            run()

