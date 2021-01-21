#!/usr/bin/env python3

""" 
read command line arguments in this following order:
    0      1    2     3         4           5            6           7        8
./run.py mode model symbol date1(start) date2(end) learning_rate iteration backtest
"""

import os
import sys

from lib import *

mode   = sys.argv[1]
model  = sys.argv[2]
symbol = sys.argv[3]
date1  = sys.argv[4]

model_path = "./models/" + model

def init():
    # initialize required root and model file system
    required = [
        "./temp", "./data", "./models", model_path, model_path + "/layers", model_path + "/kernels", model_path + "/dnn", model_path + "/res"
    ]
    for d in required:
        if os.path.exists(d) != True:
            os.mkdir(d)
    # build Futures
    os.system("./scripts/launch make")

def train():
    date2         = sys.argv[5]
    learning_rate = sys.argv[6] # needs conversion to float
    iteration     = sys.argv[7] # needs conversion to unsigned int
    backtest      = sys.argv[8] # needs conversion to unsigned int (0 or 1)
    # download, process, and save financial time series 
    dataset = process_timeseries(symbol, date1, date2, True)
    # execute shell script to run Futures (built in C++)
    # Futures will read the written dataset (in binary) and encode the dataset
    # after encoding, "dnn.py" will be called from Futures to train the deep neural network (TensorFlow)
    #os.system("./scripts/launch")

if __name__ == "__main__":
    init()
    if sys.argv[1] == "train":
        train()
