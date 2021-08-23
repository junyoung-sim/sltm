#!/usr/bin/env python3

import os, sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

model = sys.argv[1]
path  = "./models/{}/res/" .format(model)
date  = datetime.today().strftime("%Y-%m-%d")

yhat = []
with open("{}pred" .format(path), "r") as f:
    yhat = np.array([float(val) for val in f.readline().split(" ")])

plt.title("{} [{}]" .format(model, date))
plt.plot(yhat, color="red")
plt.savefig("{}prediction/{}.png" .format(path, date))

os.system("rm {}pred" .format(path))
with open("{}npy/{}.npy" .format(path, date), "wb") as f:
    np.save(f, yhat)

