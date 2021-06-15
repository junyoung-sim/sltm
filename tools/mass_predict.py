
import os, sys
from lib import *
import matplotlib.pyplot as plt

model = sys.argv[1]
start = sys.argv[2]

raw = HistoricalData(model, "2019-01-01")

data = []
dates = raw["dates"][raw["dates"].index(start):]
for i in range(len(dates)):
    data.append(mavg(raw["price"][i-171:i], 50))

with open("./temp/input", "w+") as f:
    for i in range(len(data)):
        for val in data[i]:
            f.write(str(val) + " ")
        if i != len(data) - 1:
            f.write("\n")

os.system("./encoder {}" .format(model))

encoded = []
with open("./temp/encoded", "r") as f:
    for line in f.readlines():
        encoded.append([float(val) for val in line.split(" ")])

predictor = DeepNeuralNetwork("./models/{}" .format(model))
results = predictor.run(encoded)

for i in range(results.shape[0]):
    with open("./models/{}/res/npy/{}.npy" .format(model, dates[i]), "wb") as f:
        np.save(f, results[i])
    fig = plt.figure()
    plt.plot(results[i], color="red")
    plt.savefig("./models/{}/res/prediction/{}.png" .format(model, dates[i]))
