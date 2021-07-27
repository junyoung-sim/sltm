#!/usr/bin/env python3

import sys
import numpy as np
from lib import mse
import matplotlib.pyplot as plt

model = sys.argv[1]
plot_true = int(sys.argv[2])

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

success = []
for i in range(actual.shape[0]):
    error = mse(actual[i], backtest[i])
    if error < 0.06:
        success.append(1)
        if plot_true == 1:
            plt.figure()
            plt.plot(actual[i], color="green")
            plt.plot(backtest[i], color="red")
            plt.savefig("./models/{}/eval/{} [{}].png" .format(model, i, round(error, 4)))
    else:
        success.append(0)

if plot_true == 1:
    # accurate prediction pulse
    plt.figure(figsize=(12,5))
    plt.plot(success)
    plt.savefig("./models/{}/eval/success.png" .format(model))

