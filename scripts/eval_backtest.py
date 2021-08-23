#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from data import mse, normalize

model = sys.argv[1]

actual = []
with open("./models/{}/backtest/actual" .format(model), "r") as f:
    for line in f.readlines():
        actual.append([float(val) for val in line.split(" ")])

backtest = []
with open("./models/{}/backtest/backtest" .format(model), "r") as f:
    for line in f.readlines():
        backtest.append([float(val) for val in line.split(" ")])

actual, backtest = np.array(actual), np.array(backtest)
cost = mse(actual.flatten(), backtest.flatten())

good = []
for i in range(actual.shape[0]):
    if (mse(normalize(actual[i][:10]), normalize(backtest[i][:10])) < cost) & (mse(actual[i], backtest[i]) < cost):
        plt.figure()
        plt.plot(actual[i], color="green")
        plt.plot(backtest[i], color="red")
        plt.savefig("./models/{}/backtest/test{}.png" .format(model, i))
        good.append(i)

#plt.plot([1 if i in good else 0 for i in range(actual.shape[0])])
#plt.show()

predicted = set([])
for i in good:
    for k in range(i, i+50):
        predicted.add(k)

print("Correct Predictions = {}" .format(len(good)))
print("Predicted Days = {}% ({} days)" .format(len(predicted) * 100 / (actual.shape[0] + 50), len(predicted)))
