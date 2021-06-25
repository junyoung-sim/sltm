#!/usr/bin/env python3

import sys
import numpy as np
from lib import vector_analysis
import matplotlib.pyplot as plt

model = sys.argv[1]
threshold = float(sys.argv[2])

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

correctness = 0
for n in range(actual.shape[0]):
    if vector_analysis(actual[n], backtest[n]) > threshold:
        fig = plt.figure()
        plt.plot(actual[n], color="green")
        plt.plot(backtest[n], color="red")
        plt.savefig("../etc/{}.png" .format(n))
