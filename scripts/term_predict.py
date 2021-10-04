#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data import HistoricalData, mavg, normalize

model = sys.argv[1]
start = sys.argv[2]

raw = HistoricalData(model, "2000-01-01")
price = raw["price"]
dates = raw["dates"]

data = []
for i in range(dates.index(start), len(dates)):
    data.append(normalize(mavg(price[i-169:i+1], 50)))

dates = dates[dates.index(start):]

for i in range(len(data)):
    with open("./temp/input", "w+") as f:
        for val in data[i]:
            f.write("{} " .format(val))
    os.system("./sltm run {}" .format(model))

    with open("./models/{}/res/pred" .format(model), "r") as f:
        out = [float(val) for val in f.readline().split(" ")]
    os.system("rm ./models/{}/res/pred" .format(model))

    plt.figure()
    plt.plot(out, color="red")
    plt.savefig("./models/{}/res/prediction/{}.png" .format(model, dates[i]))

    with open("./models/{}/res/npy/{}.npy" .format(model, dates[i]), "wb") as f:
        np.save(f, out)
