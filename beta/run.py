#!/usr/bin/env python3

""" 
read command line arguments in this following order:
    0      1    2     3        4         5           6           7        8
./run.py mode model symbol start_date end_date learning_rate iteration backtest
"""

import os
import sys

from lib import *

def init():
    # initialize required file system
    required = ["./temp", "./data", "./models"]
    for d in required:
        if os.path.exists(d) != True:
            os.mkdir(d)
    # build Futures
    os.system("./scripts/launch make")

def train():
    symbol, start, end = sys.argv[3], sys.argv[4], sys.argv[5]
    learning_rate, iteration, backtest = float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    dataset = process_timeseries(symbol, start, end, True) # downloads and saves processed time series
    # execute shell script to run Futures (built in C++)
    # Futures will read the written dataset (in binary) and encode the dataset
    # after encoding, "dnn.py" will be called from Futures to train the deep neural network (TensorFlow)
    #os.system("./scripts/futures")

if __name__ == "__main__":
    init()
    if sys.argv[1] == "train":
        train()
