#!/usr/bin/env python3

import sys
import numpy as np
from lib import mse
import matplotlib.pyplot as plt

model = sys.argv[1]

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

threshold = mse(actual.flatten(), backtest.flatten())

mean_profit = []
mean_loss = []

for n in range(actual.shape[0]):
    #derivative = [abs(actual[n][i+1] - actual[n][i]) for i in range(actual[n].shape[0] - 1)] # long and short
    derivative = [actual[n][i+1] - actual[n][i] for i in range(actual[n].shape[0] - 1) if actual[n][i+1] - actual[n][i] > 0] # long
    if mse(actual[n], backtest[n]) < threshold:
        mean_profit.append(sum(derivative))
        fig = plt.figure()
        plt.plot(actual[n], color="green")
        plt.plot(backtest[n], color="red")
        plt.savefig("../eval/{}.png" .format(n))
    else:
        mean_loss.append(sum(derivative))

profit_prob = round(len(mean_profit) / actual.shape[0], 8)
loss_prob = round(len(mean_loss) / actual.shape[0], 8)

mean_profit = round(sum(mean_profit) / len(mean_profit), 8)
mean_loss = round(sum(mean_loss) / len(mean_loss), 8)

tx = round(mean_profit * profit_prob, 8)
fx = round(mean_loss * loss_prob, 8)

expectation = round(tx - fx, 8)

print("\nMSE Threshold: {}" .format(threshold))

print("\nMean Profit Derivative Sum: {}" .format(mean_profit))
print("Profit Probability: {}" .format(profit_prob))

print("\nMean Loss Derivative Sum: {}" .format(mean_loss))
print("Loss Probability: {}" .format(loss_prob))

print("\nT(X) = {}" .format(tx))
print("F(X) = {}" .format(fx))

print("\nE(X) = {}" .format(expectation))
