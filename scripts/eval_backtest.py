#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from data import mse, normalize

model = sys.argv[1]
threshold = float(sys.argv[2])

actual = []
with open("./models/{}/backtest/actual" .format(model), "r") as f:
    for line in f.readlines():
        actual.append([float(val) for val in line.split(" ")])

backtest = []
with open("./models/{}/backtest/backtest" .format(model), "r") as f:
    for line in f.readlines():
        backtest.append([float(val) for val in line.split(" ")])

actual, backtest = np.array(actual), np.array(backtest)
#cost = mse(actual.flatten(), backtest.flatten())

good = []
for i in range(actual.shape[0]):
    if mse(actual[i], backtest[i]) <= threshold:
        good.append(i)
        plt.figure()
        plt.plot(actual[i], color="green")
        plt.plot(backtest[i], color="red")
        plt.savefig("./models/{}/backtest/test{}.png" .format(model, i))

signal = [1 if i in good else 0 for i in range(actual.shape[0])]
plt.figure(figsize=(12,4))
plt.plot(signal, color="blue")
plt.savefig("./models/{}/backtest/predicted_signal.png" .format(model))

predicted = set([])
for i in good:
    for k in range(i, i+50):
        predicted.add(k)
predicted = list(predicted)

print("Predictions w/ MSE lower than training MSE = {} samples = {}%" .format(len(good), len(good) * 100 / actual.shape[0]))
print("Predicted days: {} days = {}%" .format(len(predicted), len(predicted) * 100 / (actual.shape[0] + 50)))
