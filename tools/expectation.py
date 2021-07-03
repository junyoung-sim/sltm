#!/usr/bin/env python3

import sys
import numpy as np
from lib import mse
import matplotlib.pyplot as plt

model = sys.argv[1]

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

threshold = mse(actual.flatten(), backtest.flatten())

mean_good = []
mean_bad = []

for n in range(actual.shape[0]):
    #derivative = [abs(actual[n][i+1] - actual[n][i]) for i in range(actual[n].shape[0] - 1)] # long and short
    derivative = [actual[n][i+1] - actual[n][i] for i in range(actual[n].shape[0] - 1) if actual[n][i+1] - actual[n][i] > 0] # long
    if mse(actual[n], backtest[n]) < threshold:
        mean_good.append(sum(derivative))
        #fig = plt.figure()
        #plt.plot(actual[n], color="green")
        #plt.plot(backtest[n], color="red")
        #plt.savefig("../etc/{}.png" .format(n))
    else:
        mean_bad.append(sum(derivative))

good_prob = round(len(mean_good) / actual.shape[0], 8)
bad_prob = round(len(mean_bad) / actual.shape[0], 8)

mean_good = round(sum(mean_good) / len(mean_good), 8)
mean_bad = round(sum(mean_bad) / len(mean_bad), 8)

tx = round(mean_good * good_prob, 8)
fx = round(mean_bad * bad_prob, 8)

expectation = round(tx - fx, 8)

print("\nMSE Threshold: {}" .format(threshold))

print("\nMean Good Derivative Sum: {}" .format(mean_good))
print("Good Probability: {}" .format(good_prob))

print("\nMean Bad Derivative Sum: {}" .format(mean_bad))
print("Bad Probability: {}" .format(bad_prob))

print("\nT(X) = {}" .format(tx))
print("F(X) = {}" .format(fx))

print("\nE(X) = {}" .format(expectation))
