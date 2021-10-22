#!/usr/bin/env python3

import sys
import numpy as np
from data import mse
import matplotlib.pyplot as plt

training_mse = float(sys.argv[1])

actual = []
backtest = []

with open("./models/spy-paper-version/backtest/actual", "r") as f1, open("./models/spy-paper-version/backtest/backtest", "r") as f2:
    for line in f1.readlines():
        actual.append([float(val) for val in line.split(" ")])
    for line in f2.readlines():
        backtest.append([float(val) for val in line.split(" ")])

actual = np.array(actual)
backtest = np.array(backtest)

mse_dist = []
for i in range(actual.shape[0]):
    mse_dist.append(mse(actual[i], backtest[i]))

n, bins, patches = plt.hist(mse_dist, bins=100, edgecolor="black")

for i in range(100):
    if bins[i] < training_mse:
        patches[i].set_fc('g')
    else:
        patches[i].set_fc('r')

plt.show()
