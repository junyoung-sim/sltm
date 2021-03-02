#!/usr/bin/env python3

""" In-depth analysis of individual trend models  """

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from lib import smoothing

model = sys.argv[1]
date  = sys.argv[2]

prediction = np.load("./models/" + model + "/res/npy/" + date + ".npy")
derivative = smoothing([prediction[i+1] - prediction[i] for i in range(prediction.shape[0] - 1)])
# identify when direction of derivative changes (extreme point)
extreme_point = [i for i in range(len(derivative) - 1) if abs(derivative[i+1])/derivative[i+1] != abs(derivative[i])/derivative[i]]

date = datetime.strptime(date, "%Y-%m-%d")
for p in extreme_point:
    print("Extreme Point @ #{} = {}" .format(p, str(date + timedelta(days=p))[:10]))
