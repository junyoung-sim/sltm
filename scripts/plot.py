#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

model = sys.argv[1]
date  = sys.argv[2]
path  "./models/" + model + "/res/" + date

yhat = []
with open(path, "r") as f:
    yhat = [float(val) for val in f.readline()]

plt.title("{} [{}]" .format(model, date))
plt.plot(yhat, color="red")
plt.savefig(path + ".png")
