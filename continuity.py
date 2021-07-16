#!/usr/bin/env python3

import sys
import numpy as np
from lib import mse
import matplotlib.pyplot as plt

model = sys.argv[1]

actual = np.load("./models/{}/backtest/actual.npy" .format(model))
backtest = np.load("./models/{}/backtest/backtest.npy" .format(model))

error = [mse(actual[i], backtest[i]) for i in range(actual.shape[0])]

success = []
for i in range(len(error)):
    if error[i] < 0.06:
        success.append(1)
        plt.figure()
        plt.plot(actual[i], color="green")
        plt.plot(backtest[i], color="red")
        plt.savefig("./models/{}/eval/{} [{}].png" .format(model, i, round(error[i], 4)))
    else:
        success.append(0)

plt.figure(figsize=(12,5))
plt.plot(success)
plt.savefig("./models/{}/eval/success.png" .format(model))

observed = set([])
observable_range = actual.shape[0] + actual.shape[1]
arg_success = [i for i in range(len(error)) if error[i] < 0.06]
for p in arg_success:
    for i in range(p, p+actual.shape[1]):
        observed.add(i)
observed = list(observed)

observed = [1 if i in observed else 0 for i in range(observable_range)]
plt.figure(figsize=(12,5))
plt.plot(observed)
plt.savefig("./models/{}/eval/observed.png" .format(model))

#observed_horizons = []
#for n in range(0, actual.shape[0] - 75, 75):
#    predicted = False
#    for i in range(n, n + 75):
#        predicted = error[i] < 0.06
#        if predicted:
#            break
#    if predicted:
#        observed_horizons.append(1)
#    else:
#        observed_horizons.append(0)

#plt.plot(observed_horizons)
#plt.show()
