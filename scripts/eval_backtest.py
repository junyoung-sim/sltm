#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

model = sys.argv[1]

actual = []
with open("./models/{}/backtest/actual" .format(model), "r") as f:
    for line in f.readlines():
        actual.append([float(val) for val in line.split(" ")])

backtest = []
with open("./models/{}/backtest/backtest" .format(model), "r") as f:
    for line in f.readlines():
        backtest.append([float(val) for val in line.split(" ")])

for i in range(len(actual)):
    plt.figure()
    plt.plot(actual[i], color="green")
    plt.plot(backtest[i], color="red")
    plt.savefig("./models/{}/backtest/test{}.png" .format(model, i))
