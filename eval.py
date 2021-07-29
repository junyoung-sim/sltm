#!/usr/bin/env python3

import os
import sys
import numpy as np
from lib import mse, normalize
import matplotlib.pyplot as plt

model = sys.argv[1]
days = int(sys.argv[2])
threshold = float(sys.argv[3])
accurate = float(sys.argv[4])
plot_true = int(sys.argv[5])

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

good = []
for d in range(actual.shape[0]):
    if (mse(normalize(actual[d][:days]), normalize(backtest[d][:days])) < threshold) & (mse(actual[d], backtest[d]) < accurate):
        good.append(d)
        if plot_true == 1:
            os.system("cp ./models/{}/backtest/test{}.png ./models/{}/eval/" .format(model, d, model))

predicted = set([])
for d in good:
    for i in range(d, d+75):
        predicted.add(i)

print("\n", model)
print("Correct Predictions: {}" .format(len(good)))
print("% of Predicted Days: {}" .format(round(len(predicted) / (actual.shape[0] + 75) * 100, 2)))
print("\n")


