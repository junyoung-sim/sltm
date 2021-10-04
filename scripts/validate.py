#!/usr/bin/env python3

import os, sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from data import *

model = sys.argv[1]
path  = "./models/{}/res/" .format(model)

os.system("rm {}validation/*.png" .format(path))

raw = HistoricalData(symbol=model, start="2000-01-01")
trend = mavg(raw["price"], 10)
dates = raw["dates"][10:]

for f in os.listdir("{}npy/" .format(path)):
    if f.endswith(".npy") and f[:10] != datetime.today().strftime("%Y-%m-%d"):
        date = f[:10]
        actual = normalize(trend[dates.index(date):])
        if len(actual) < 50:
            prediction = normalize(np.load("{}npy/{}" .format(path, f))[:len(actual)])
            error = mse(actual, prediction)
            plt.figure()
            plt.title("{} [{}]" .format(model, date))
            plt.plot(actual, color="green")
            plt.plot(prediction, color="red")
            plt.savefig("{}validation/{} MSE={}.png" .format(path, date, round(error, 4)))
        elif len(actual) >= 50:
            actual = normalize(actual[:50])
            prediction = np.load("{}npy/{}" .format(path, f))
            error = mse(actual, prediction)
            plt.figure()
            plt.title("{} [{}]" .format(model, date))
            plt.plot(actual, color="green")
            plt.plot(prediction, color="red")
            plt.savefig("{}expired/{} MSE={}.png" .format(path, date, round(error, 4)))
            os.system("mv {}npy/{} {}expired" .format(path, f, path))
            os.system("rm {}prediction/{}.png" .format(path, date))
