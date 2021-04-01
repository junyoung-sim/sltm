#!/usr/bin/env python3

import numpy as np

model, date = input("MODEL / YYYY-MM-DD: ").split()
prediction = np.load("./models/" + model + "/res/npy/" + date + ".npy")

derivative = [prediction[i+1] - prediction[i] for i in range(prediction.shape[0] - 1)]
curve_point = [i for i in range(len(derivative) - 1) if abs(derivative[i]) / derivative[i] != abs(derivative[i+1]) / derivative[i+1]]
print("Curve Point Indices: ", curve_point)
