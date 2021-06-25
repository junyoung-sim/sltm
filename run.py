#!/usr/bin/env python3

""" 
    0      1    2         3           4       5      6
./run.py mode model date1(start) date2(end) epoch backtest
      required     |          only for training

<model> should be the stock symbol
<date1> and <date2> should be in YYYY-MM-DD format
<date2> can be stated as yyyy-mm-dd as an alternative for the current date
"""

import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
from lib import *

mode       = sys.argv[1]
model      = sys.argv[2]
model_path = "./models/" + model

def init():
    print("\nModel: {}" .format(model))
    okay = False
    if mode == "train":
        print("\nInitializing file system...")
        required = [
            model_path,
            model_path + "/kernels",
            model_path + "/dnn",
            model_path + "/backtest",
            model_path + "/trained-samples",
            model_path + "/res",
            model_path + "/res/npy",
            model_path + "/res/prediction",
            model_path + "/res/validation",
            model_path + "/res/expired"
        ]
        for d in required:
            if not os.path.exists(d): # check if required directories exists; if not, create them
                os.mkdir(d)
        okay = True
    elif mode == "run":
        if os.path.exists("{}/{}" .format(model_path, model)): # checking if trained model exists
            print("Model detected!")
            okay = True
        else:
            print("\nRequested model does not exist!\n")
    else:
        print("\nInvalid mode given!\n")
    if okay:
        for root, dirs, files in os.walk("./temp"): # delete files in ./temp
            for f in files:
                os.system("rm ./temp/{}" .format(f))
        print("\nBuilding the encoder...\n")
        os.system("./scripts/build")
    return okay

def train():
    date1    = sys.argv[3] # start date
    date2    = sys.argv[4] # end date
    epoch    = int(sys.argv[5])
    backtest = float(sys.argv[6]) # last n% of the dataset will be backtested
    # time series sampling
    dataset = sample_timeseries_dataset(model, date1, date2)
    # run encoder on training inputs (C executable)
    os.system("./encoder {}" .format(model))
    # read encoded inputs
    encoded = []
    with open("./temp/encoded", "r") as f:
        for line in f.readlines():
            encoded.append([float(val) for val in line.split(" ")])
    dataset["input"] = np.array(encoded)
    print("{} samples (Size = {})\n{}\n" .format(dataset["input"].shape[0], dataset["input"].shape[1], dataset["input"]))    
    # train prediction model
    predictor = DeepNeuralNetwork(model_path)
    predictor.train(dataset, epoch, backtest)
    if not os.path.exists("{}/{}" .format(model_path, model)):
        os.system("touch {}/{}" .format(model_path, model)) # create a file indicating that the prediction model is trained

def run():
    sample_recent_input(model) # process recent D-121 MAVG 50 input
    # run encoder on input
    os.system("./encoder " + model)
    # read encoded input
    encoded = []
    with open("./temp/encoded", "r") as f:
        encoded = np.array([[float(val) for val in f.readline().split(" ")]])
    print("\nEncoded input:\n", encoded)
    # run prediction model
    predictor = DeepNeuralNetwork(model_path)
    result = predictor.run(encoded)[0]
    print("Model Prediction:\n", np.array(result), "\n")
    # save prediction results
    plt.plot(result, color="red")
    plt.savefig("{}/res/prediction/{}.png" .format(model_path, datetime.today().strftime("%Y-%m-%d")))
    with open("{}/res/npy/{}.npy" .format(model_path, datetime.today().strftime("%Y-%m-%d")), "wb") as f:
        np.save(f, result)
    # validate trend models
    validation(model)

if __name__ == "__main__":
    if init():
        eval(mode + "()")

