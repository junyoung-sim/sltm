
import os
import numpy as np
import tqdm as tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

def normalize(data=[]):
    return [(p - min(data)) / (max(data) - min(data)) for p in data]

def mavg(data=[], window=int()):
    return [sum(data[i:i+window]) / window for i in range(0, len(data) - window)]

def mse(actual=[], prediction=[]):
    return sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)

def RealtimePrice(symbol=""):
    price = list(DataReader(symbol, "yahoo", datetime.today().strftime("%Y-%m-%d"))["Adj Close"])[0]
    return price

def HistoricalData(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    if end != "yyyy-mm-dd":
        download = DataReader(symbol, "yahoo", start, end)
    else:
        download = DataReader(symbol, "yahoo", start)
    download.to_csv("./data/" + symbol + ".csv")
    price = list(download["Adj Close"]) # adjusted close price
    with open("./data/" + symbol + ".csv", "r") as f:
         dates = [line[:10] for line in f.readlines()] # get date of each price
         del dates[0] # first line is category header
    return {"price": price, "dates": dates}

def generate_timeseries_dataset(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    training_input, training_output = [], []
    raw = HistoricalData(symbol, start, end)
    price, dates = raw["price"], raw["dates"]
    # sample time series dataset
    loop = tqdm.tqdm(total=len(price)-206, position=0, leave=False)
    for i in range(len(price)-206):
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        training_input.append(normalize(mavg(price[i:i+171], 50))) # D-121 MAVG 50 input
        training_output.append(normalize(mavg(price[i+121:i+206], 10))) # D+75 MAVG 10 output
        loop.update(1)
    training_input, training_output = np.array(training_input), np.array(training_output)
    with open("./temp/input", "w+") as f:
        for i in range(training_input.shape[0]): # write each input into each line
            for val in training_input[i]:
                f.write(str(val) + " ")
            if i != training_input.shape[0] - 1:
                f.write("\n")
    return {"input": training_input, "output": training_output}

def validation(model_path=""):
    for f in os.listdir(model_path + "/res/npy"):
        if f.endswith(".npy") and f[:10] != datetime.today().strftime("%Y-%m-%d"):
            date, symbol = f[:10], f[11:-4]
            raw = HistoricalData(symbol, "2021-01-01")
            trend, dates = mavg(raw["price"], 10), raw["dates"][10:]
            # validate trend models that are at least 3 days old
            actual = normalize(trend[dates.index(date):])
            if len(actual) >= 3:
                prediction = normalize(np.load("{}/res/npy/{}" .format(model_path, f))[:len(actual)])
                error = mse(actual, prediction) # mean squared error
                # output validation results
                print("{}-{} @D+{}: MSE = {}" .format(symbol, date, len(actual), error))
                fig = plt.figure()
                plt.plot(actual, color="green")
                plt.plot(prediction, color="red")
                plt.show()

