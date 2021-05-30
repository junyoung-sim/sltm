#!/usr/bin/env python3

""" 
Intraday trading model using Futures.0
Run this script during the same day before the next market open.
"""

import os, sys
from time import time, sleep
from datetime import datetime
import matplotlib.pyplot as plt

from lib import *

model      = sys.argv[1]
model_path = "./models/" + model

def refresh(seconds=int()):
    sleep(seconds)
    os.system("clear")

def run():
    # wait for market open
    market_open = datetime.strptime(str(datetime.now())[:10] + " 09:30:00", "%Y-%m-%d %H:%M:%S")
    while True:
        now = datetime.now()
        if datetime.strftime(now, "%Y-%m-%d %H:%M:%S") != str(market_open):
            time_left = str(market_open - now)[:-7] # omit microseconds
            print("Waiting for market open... ({} left)" .format(time_left))
        else:
            break
        refresh(seconds=1)
    # sample intraday 1 minute price data
    delay = 0
    intraday = []
    market_close = datetime.strptime(str(datetime.now())[:10] + " 16:00:00", "%Y-%m-%d %H:%M:%S")
    while True:
        now = datetime.now()
        if datetime.strftime(now, "%Y-%m-%d %H:%M:%S") != str(market_close):
            start = time()
            intraday.append(RealtimePrice(model))
            end = time()
            delay = end - start
            print("{} @{}: ${}" .format(model, datetime.strftime(now, "%Y-%m-%d %H:%M:%S"), intraday[-1]))
            sleep(60 - delay)
        else:
            break
    # save the data of the last 120 minutes in ./temp/input
    with open("./temp/input", "w+") as f:
        for val in normalize(mavg(intraday[-171:], 50)):
            f.write(str(val) + " ")
    # run encoder
    system("./encoder " + model)
    # read encoded data
    with open("./temp/encoded", "r") as f:
        encoded = [[float(val) for val in f.readline().split(" ")]]
    print("\nEncoded Intrady Price MAVG:\n", encoded)
    # run prediction model
    predictor = DeepNeuralNetwork(model_path)
    result = predictor.run(encoded)[0]
    print("Model Prediction:\n{}\n" .format(result))
    # show prediction plot
    plt.plot(result, color="red")
    plt.show()

if __name__ == "__main__":
    run()

