#!/usr/bin/env python3

import os, sys
import numpy as np
from datetime import datetime
from data import *

model = sys.argv[1]
path  = "./models/{}/res/" .format(model)

raw = HistoricalData(symbol=model, start="2021-01-01")
trend = mavg(raw["price"], 10)
dates = raw["dates"][10:]

for f in os.listdir(path):
    if f.endswith(".npy") and f[:10] != datetime.today().strftime("%Y-%m-%d"):
        date = f[:10]
        actual = normalize(trend[dates.index(date):])
        if len(actual) <= 75:
            prediction = normalize(np.load("{}{}" .format(path, f))[:len(actual)])
            error = mse(actual, prediction)
            plt.figure()
            plt.title("{} [{} D+{}]" .format(model, date, len(actual)))
            plt.plot(actual, color="green")
            plt.plot(prediction, color="red")
            plt.savefig("{}{} MSE: {}.png" .format(path, date, error))
